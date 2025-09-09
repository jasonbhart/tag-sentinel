"""Page session orchestration for coordinated browser capture.

This module provides the PageSession class that coordinates all capture
components (network observer, console observer, cookie collector, pre-steps
executor) to perform complete page capture with configurable wait strategies
and error recovery.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from .network_observer import NetworkObserver
from .console_observer import CombinedObserver
from .cookie_collector import CookieCollector
from .presteps import PreStepsExecutor
from ..models.capture import (
    PageResult, 
    CaptureStatus, 
    ArtifactPaths
)

logger = logging.getLogger(__name__)


class WaitStrategy:
    """Available wait strategies for page load completion."""
    NETWORKIDLE = "networkidle"
    LOAD = "load" 
    DOMCONTENTLOADED = "domcontentloaded"
    SELECTOR = "selector"
    TIMEOUT = "timeout"
    CUSTOM = "custom"


class PageSessionConfig:
    """Configuration for page capture sessions."""
    
    def __init__(
        self,
        wait_strategy: str = WaitStrategy.NETWORKIDLE,
        wait_timeout_ms: int = 30000,
        wait_selector: Optional[str] = None,
        wait_custom_js: Optional[str] = None,
        pre_steps: Optional[List[Dict[str, Any]]] = None,
        enable_network_capture: bool = True,
        enable_console_capture: bool = True,
        enable_cookie_capture: bool = True,
        redact_cookie_values: bool = True,
        filter_console_noise: bool = True,
        capture_response_bodies: bool = False,
        max_response_size: int = 50000,
        artifacts_dir: Optional[Path] = None,
        take_screenshot: bool = False,
        screenshot_on_error: bool = True,
        enable_har: bool = False,
        enable_trace: bool = False,
        user_agent_override: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize page session configuration.
        
        Args:
            wait_strategy: Strategy for determining page load completion
            wait_timeout_ms: Maximum time to wait for page load
            wait_selector: CSS selector to wait for (when using selector strategy)
            wait_custom_js: JavaScript condition to wait for (when using custom strategy)
            pre_steps: List of pre-capture actions to execute
            enable_network_capture: Whether to capture network requests
            enable_console_capture: Whether to capture console messages
            enable_cookie_capture: Whether to capture cookies
            redact_cookie_values: Whether to redact cookie values for privacy
            filter_console_noise: Whether to filter noisy console messages
            capture_response_bodies: Whether to capture response bodies
            max_response_size: Maximum response body size to capture
            artifacts_dir: Directory for debug artifacts
            take_screenshot: Whether to take page screenshot
            screenshot_on_error: Whether to take screenshot on errors
            enable_har: Whether to generate HAR file
            enable_trace: Whether to generate Playwright trace
            user_agent_override: Custom User-Agent string
            extra_headers: Additional HTTP headers
        """
        self.wait_strategy = wait_strategy
        self.wait_timeout_ms = wait_timeout_ms
        self.wait_selector = wait_selector
        self.wait_custom_js = wait_custom_js
        self.pre_steps = pre_steps or []
        self.enable_network_capture = enable_network_capture
        self.enable_console_capture = enable_console_capture
        self.enable_cookie_capture = enable_cookie_capture
        self.redact_cookie_values = redact_cookie_values
        self.filter_console_noise = filter_console_noise
        self.capture_response_bodies = capture_response_bodies
        self.max_response_size = max_response_size
        self.artifacts_dir = artifacts_dir
        self.take_screenshot = take_screenshot
        self.screenshot_on_error = screenshot_on_error
        self.enable_har = enable_har
        self.enable_trace = enable_trace
        self.user_agent_override = user_agent_override
        self.extra_headers = extra_headers or {}


class PageSession:
    """Orchestrates complete page capture with all observers and components."""
    
    def __init__(self, page: Page, config: PageSessionConfig):
        """Initialize page session.
        
        Args:
            page: Playwright page for capture
            config: Page session configuration
        """
        self.page = page
        self.config = config
        self.page_result: Optional[PageResult] = None
        
        # Initialize components based on configuration
        self.network_observer: Optional[NetworkObserver] = None
        self.console_observer: Optional[CombinedObserver] = None
        self.cookie_collector: Optional[CookieCollector] = None
        self.presteps_executor: Optional[PreStepsExecutor] = None
        
        # Session state
        self.session_start_time: Optional[datetime] = None
        self.navigation_start_time: Optional[datetime] = None
        self.load_complete_time: Optional[datetime] = None
        self._callbacks: List[Callable[[PageResult], None]] = []
        
        # Initialize components
        self._initialize_components()
    
    def add_callback(self, callback: Callable[[PageResult], None]) -> None:
        """Add callback to be called when capture completes.
        
        Args:
            callback: Function to call with completed PageResult
        """
        self._callbacks.append(callback)
    
    def _initialize_components(self) -> None:
        """Initialize capture components based on configuration."""
        if self.config.enable_network_capture:
            self.network_observer = NetworkObserver(self.page)
            
        if self.config.enable_console_capture:
            self.console_observer = CombinedObserver(
                self.page, 
                filter_console_noise=self.config.filter_console_noise
            )
            
        if self.config.enable_cookie_capture:
            # Cookie collector needs browser context, will be initialized during capture
            pass
            
        if self.config.pre_steps:
            self.presteps_executor = PreStepsExecutor(
                self.page,
                timeout_ms=self.config.wait_timeout_ms,
                retry_count=3
            )
        
        logger.debug("Page session components initialized")
    
    async def capture_page(self, url: str) -> PageResult:
        """Perform complete page capture.
        
        Args:
            url: URL to capture
            
        Returns:
            Complete PageResult with all captured data
        """
        self.session_start_time = datetime.utcnow()
        
        # Initialize page result
        self.page_result = PageResult(
            url=url,
            capture_time=self.session_start_time,
            capture_status=CaptureStatus.FAILED  # Will be updated on success
        )
        
        try:
            # Set custom headers if configured
            if self.config.extra_headers:
                await self.page.set_extra_http_headers(self.config.extra_headers)
            
            # Execute pre-steps if configured
            if self.presteps_executor and self.config.pre_steps:
                logger.info("Executing pre-steps")
                presteps_success = await self.presteps_executor.execute_steps(self.config.pre_steps)
                if not presteps_success:
                    logger.warning("Some pre-steps failed, continuing with capture")
            
            # Initialize cookie collector with context
            if self.config.enable_cookie_capture:
                self.cookie_collector = CookieCollector(
                    self.page.context,
                    redact_values=self.config.redact_cookie_values
                )
            
            # Navigate to page
            await self._navigate_to_page(url)
            
            # Wait for page load completion
            await self._wait_for_load_completion()
            
            # Finalize observers
            await self._finalize_capture()
            
            # Generate artifacts if configured
            await self._generate_artifacts()
            
            # Update capture status
            self.page_result.capture_status = CaptureStatus.SUCCESS
            
            logger.info(f"Page capture completed successfully: {url}")
            
        except Exception as e:
            logger.error(f"Page capture failed: {e}")
            self.page_result.capture_error = str(e)
            
            # Determine specific failure type
            if isinstance(e, PlaywrightTimeoutError):
                self.page_result.capture_status = CaptureStatus.TIMEOUT
            else:
                self.page_result.capture_status = CaptureStatus.FAILED
            
            # Take error screenshot if configured
            if self.config.screenshot_on_error:
                await self._take_error_screenshot()
        
        # Calculate load time
        if self.navigation_start_time and self.load_complete_time:
            load_time = (self.load_complete_time - self.navigation_start_time).total_seconds() * 1000
            self.page_result.load_time_ms = load_time
        
        # Call callbacks
        for callback in self._callbacks:
            try:
                callback(self.page_result)
            except Exception as e:
                logger.error(f"Error in page session callback: {e}")
        
        return self.page_result
    
    async def _navigate_to_page(self, url: str) -> None:
        """Navigate to the target page.
        
        Args:
            url: URL to navigate to
        """
        self.navigation_start_time = datetime.utcnow()
        
        try:
            response = await self.page.goto(
                url, 
                timeout=self.config.wait_timeout_ms,
                wait_until="commit"  # Don't wait for full load here
            )
            
            if response:
                # Update final URL in case of redirects
                self.page_result.final_url = response.url
                
                # Get page title
                try:
                    self.page_result.title = await self.page.title()
                except Exception as e:
                    logger.debug(f"Failed to get page title: {e}")
            
            logger.debug(f"Navigation completed: {url}")
            
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            raise
    
    async def _wait_for_load_completion(self) -> None:
        """Wait for page load completion based on configured strategy."""
        try:
            if self.config.wait_strategy == WaitStrategy.NETWORKIDLE:
                await self.page.wait_for_load_state(
                    "networkidle", 
                    timeout=self.config.wait_timeout_ms
                )
                
            elif self.config.wait_strategy == WaitStrategy.LOAD:
                await self.page.wait_for_load_state(
                    "load", 
                    timeout=self.config.wait_timeout_ms
                )
                
            elif self.config.wait_strategy == WaitStrategy.DOMCONTENTLOADED:
                await self.page.wait_for_load_state(
                    "domcontentloaded", 
                    timeout=self.config.wait_timeout_ms
                )
                
            elif self.config.wait_strategy == WaitStrategy.SELECTOR:
                if not self.config.wait_selector:
                    raise ValueError("wait_selector required for selector strategy")
                await self.page.wait_for_selector(
                    self.config.wait_selector,
                    timeout=self.config.wait_timeout_ms
                )
                
            elif self.config.wait_strategy == WaitStrategy.TIMEOUT:
                await self.page.wait_for_timeout(self.config.wait_timeout_ms)
                
            elif self.config.wait_strategy == WaitStrategy.CUSTOM:
                if not self.config.wait_custom_js:
                    raise ValueError("wait_custom_js required for custom strategy")
                await self.page.wait_for_function(
                    self.config.wait_custom_js,
                    timeout=self.config.wait_timeout_ms
                )
            
            self.load_complete_time = datetime.utcnow()
            logger.debug(f"Load completion detected: {self.config.wait_strategy}")
            
        except PlaywrightTimeoutError:
            logger.warning(f"Load wait timeout ({self.config.wait_strategy})")
            self.load_complete_time = datetime.utcnow()
            # Don't re-raise, continue with partial capture
            self.page_result.capture_status = CaptureStatus.PARTIAL
    
    async def _finalize_capture(self) -> None:
        """Finalize all observers and collect captured data."""
        # Finalize network observer
        if self.network_observer:
            self.network_observer.finalize_pending_requests()
            self.page_result.network_requests = self.network_observer.get_completed_requests()
        
        # Collect console logs
        if self.console_observer:
            self.page_result.console_logs = self.console_observer.get_console_logs()
            self.page_result.page_errors = self.console_observer.get_page_errors()
        
        # Collect cookies
        if self.cookie_collector:
            cookies = await self.cookie_collector.collect_cookies(self.page_result.url)
            self.page_result.cookies = cookies
        
        # Add performance metrics
        try:
            # Get navigation timing data
            timing = await self.page.evaluate('''
                () => {
                    const navigation = performance.getEntriesByType('navigation')[0];
                    return navigation ? {
                        dns_lookup: navigation.domainLookupEnd - navigation.domainLookupStart,
                        connect_time: navigation.connectEnd - navigation.connectStart,
                        request_time: navigation.responseStart - navigation.requestStart,
                        response_time: navigation.responseEnd - navigation.responseStart,
                        dom_interactive: navigation.domInteractive - navigation.navigationStart,
                        dom_complete: navigation.domComplete - navigation.navigationStart,
                        load_event: navigation.loadEventEnd - navigation.navigationStart,
                    } : {};
                }
            ''')
            
            self.page_result.metrics.update(timing)
            
        except Exception as e:
            logger.debug(f"Failed to collect performance metrics: {e}")
        
        logger.debug("Capture finalization completed")
    
    async def _generate_artifacts(self) -> None:
        """Generate debug artifacts if configured."""
        if not self.config.artifacts_dir:
            return
        
        artifacts_dir = Path(self.config.artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        artifacts = ArtifactPaths()
        url_safe = urlparse(self.page_result.url).netloc.replace(':', '_')
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{url_safe}_{timestamp}"
        
        try:
            # Take screenshot if configured
            if self.config.take_screenshot:
                screenshot_path = artifacts_dir / f"{base_filename}.png"
                await self.page.screenshot(path=screenshot_path, full_page=True)
                artifacts.screenshot_file = screenshot_path
                logger.debug(f"Screenshot saved: {screenshot_path}")
            
            # Save page source
            try:
                page_source = await self.page.content()
                source_path = artifacts_dir / f"{base_filename}.html"
                source_path.write_text(page_source, encoding='utf-8')
                artifacts.page_source = source_path
                logger.debug(f"Page source saved: {source_path}")
            except Exception as e:
                logger.debug(f"Failed to save page source: {e}")
            
            # Generate HAR if configured
            if self.config.enable_har:
                # HAR generation would be handled by browser context configuration
                har_path = artifacts_dir / f"{base_filename}.har"
                if har_path.exists():
                    artifacts.har_file = har_path
                    logger.debug(f"HAR file found: {har_path}")
            
            # Trace file if configured
            if self.config.enable_trace:
                # Trace would be handled by browser context configuration  
                trace_path = artifacts_dir / f"{base_filename}.zip"
                if trace_path.exists():
                    artifacts.trace_file = trace_path
                    logger.debug(f"Trace file found: {trace_path}")
        
        except Exception as e:
            logger.error(f"Error generating artifacts: {e}")
        
        if artifacts.has_artifacts:
            self.page_result.set_artifacts(artifacts)
    
    async def _take_error_screenshot(self) -> None:
        """Take screenshot on error if configured."""
        if not self.config.artifacts_dir:
            return
        
        try:
            artifacts_dir = Path(self.config.artifacts_dir)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            url_safe = urlparse(self.page_result.url).netloc.replace(':', '_')
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            error_screenshot_path = artifacts_dir / f"{url_safe}_{timestamp}_error.png"
            
            await self.page.screenshot(path=error_screenshot_path, full_page=True)
            
            # Update artifacts
            if not self.page_result.artifacts:
                self.page_result.artifacts = ArtifactPaths()
            self.page_result.artifacts.screenshot_file = error_screenshot_path
            
            logger.info(f"Error screenshot saved: {error_screenshot_path}")
            
        except Exception as e:
            logger.error(f"Failed to take error screenshot: {e}")
    
    def get_session_duration_ms(self) -> Optional[float]:
        """Get total session duration in milliseconds.
        
        Returns:
            Session duration or None if not complete
        """
        if self.session_start_time:
            end_time = self.load_complete_time or datetime.utcnow()
            return (end_time - self.session_start_time).total_seconds() * 1000
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        stats = {
            'session_duration_ms': self.get_session_duration_ms(),
            'capture_status': self.page_result.capture_status.value if self.page_result else 'not_started',
            'has_artifacts': bool(self.page_result and self.page_result.artifacts and self.page_result.artifacts.has_artifacts),
        }
        
        # Add component stats
        if self.network_observer:
            stats['network'] = self.network_observer.get_stats()
        
        if self.console_observer:
            stats['console'] = self.console_observer.get_stats()
        
        if self.cookie_collector:
            stats['cookies'] = self.cookie_collector.get_stats()
        
        if self.presteps_executor:
            stats['presteps'] = self.presteps_executor.get_stats()
        
        return stats
    
    def __repr__(self) -> str:
        """String representation of page session."""
        status = self.page_result.capture_status.value if self.page_result else 'not_started'
        duration = self.get_session_duration_ms()
        return (
            f"PageSession(url={self.page_result.url if self.page_result else 'none'}, "
            f"status={status}, duration={duration}ms)"
        )