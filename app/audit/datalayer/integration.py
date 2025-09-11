"""Browser integration points for dataLayer capture coordination.

This module provides integration with EPIC 2 browser capture workflow, coordinating
timing, JavaScript execution context, and browser automation for seamless
dataLayer capture alongside other auditing operations.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Awaitable
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from .models import DataLayerSnapshot, DLContext
from .config import CaptureConfig, DataLayerConfig
from .snapshot import Snapshotter
from .service import DataLayerService
from .runtime_validation import validate_types

logger = logging.getLogger(__name__)

try:
    from playwright.async_api import Page, BrowserContext
    HAS_PLAYWRIGHT = True
except ImportError:
    logger.warning("Playwright not available - browser integration will be disabled")
    Page = None
    BrowserContext = None
    HAS_PLAYWRIGHT = False


@dataclass
class BrowserIntegrationConfig:
    """Configuration for browser integration."""
    
    # Timing configuration
    wait_after_load_ms: int = 2000
    max_wait_for_datalayer_ms: int = 5000
    capture_delay_ms: int = 500
    
    # JavaScript execution settings
    inject_capture_script: bool = True
    monitor_datalayer_changes: bool = True
    capture_timing_data: bool = True
    
    # Integration settings
    coordinate_with_other_captures: bool = True
    preserve_browser_state: bool = True
    handle_spa_navigation: bool = True
    
    # Error handling
    fallback_on_capture_failure: bool = True
    max_capture_retries: int = 3
    timeout_behavior: str = "continue"  # "continue" | "fail" | "retry"


@dataclass
class CaptureContext:
    """Context information for a dataLayer capture session."""
    
    page_url: str
    page: Optional[Any] = None  # Playwright Page object
    browser_context: Optional[Any] = None  # Playwright BrowserContext
    capture_timestamp: Optional[datetime] = None
    page_load_timestamp: Optional[datetime] = None
    timing_data: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Integration state
    other_captures_completed: bool = False
    spa_navigation_detected: bool = False
    datalayer_ready: bool = False


class BrowserIntegrationError(Exception):
    """Errors during browser integration operations."""
    pass


class DataLayerBrowserIntegrator:
    """Integrates dataLayer capture with browser automation workflow."""
    
    def __init__(
        self,
        datalayer_service: DataLayerService,
        integration_config: Optional[BrowserIntegrationConfig] = None
    ):
        """Initialize browser integrator.
        
        Args:
            datalayer_service: DataLayer service instance
            integration_config: Browser integration configuration
        """
        self.datalayer_service = datalayer_service
        self.integration_config = integration_config or BrowserIntegrationConfig()
        
        # State management
        self.active_captures: Dict[str, CaptureContext] = {}
        self.capture_hooks: Dict[str, List[Callable]] = {
            'pre_capture': [],
            'post_capture': [],
            'on_error': []
        }
        
        # Performance monitoring
        self.capture_stats = {
            'total_captures': 0,
            'successful_captures': 0,
            'failed_captures': 0,
            'avg_capture_time_ms': 0.0
        }
        
        if not HAS_PLAYWRIGHT:
            logger.warning("Playwright unavailable - browser integration disabled")
    
    def register_capture_hook(
        self,
        hook_type: str,
        callback: Callable[[CaptureContext], Awaitable[None]]
    ) -> None:
        """Register a callback hook for capture events.
        
        Args:
            hook_type: Type of hook ('pre_capture', 'post_capture', 'on_error')
            callback: Async callback function
        """
        if hook_type not in self.capture_hooks:
            raise ValueError(f"Invalid hook type: {hook_type}")
        
        self.capture_hooks[hook_type].append(callback)
        logger.debug(f"Registered {hook_type} hook: {callback.__name__}")
    
    @asynccontextmanager
    async def capture_session(
        self,
        page: Any,  # Playwright Page
        page_url: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Async context manager for coordinated dataLayer capture.
        
        Args:
            page: Playwright Page object
            page_url: URL being captured
            metadata: Additional metadata for capture context
            
        Yields:
            CaptureContext for the session
        """
        if not HAS_PLAYWRIGHT:
            raise BrowserIntegrationError("Playwright not available for browser integration")
        
        # Create capture context
        context = CaptureContext(
            page_url=page_url,
            page=page,
            browser_context=page.context,
            capture_timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        session_id = f"{page_url}_{datetime.utcnow().timestamp()}"
        self.active_captures[session_id] = context
        
        try:
            # Initialize capture session
            await self._initialize_capture_session(context)
            
            # Execute pre-capture hooks
            await self._execute_hooks('pre_capture', context)
            
            yield context
            
            # Execute post-capture hooks
            await self._execute_hooks('post_capture', context)
            
        except Exception as e:
            logger.error(f"Capture session error for {page_url}: {e}")
            await self._execute_hooks('on_error', context)
            raise BrowserIntegrationError(f"Capture session failed: {e}") from e
        
        finally:
            # Cleanup
            if session_id in self.active_captures:
                del self.active_captures[session_id]
    
    async def capture_datalayer_coordinated(
        self,
        context: CaptureContext,
        wait_for_other_captures: bool = True
    ) -> Optional[DataLayerSnapshot]:
        """Capture dataLayer with coordination with other browser operations.
        
        Args:
            context: Capture context
            wait_for_other_captures: Whether to wait for other captures to complete
            
        Returns:
            DataLayer snapshot or None if capture failed
        """
        start_time = datetime.utcnow()
        
        try:
            # Wait for page stability and other captures if needed
            if wait_for_other_captures:
                await self._wait_for_capture_readiness(context)
            
            # Inject monitoring scripts if configured
            if self.integration_config.inject_capture_script:
                await self._inject_monitoring_scripts(context)
            
            # Wait for dataLayer readiness
            await self._wait_for_datalayer_ready(context)
            
            # Perform the actual capture
            dl_context = DLContext(
                page_url=context.page_url,
                capture_method="browser_integration",
                timestamp=datetime.utcnow(),
                metadata=context.metadata
            )
            
            # Use the snapshotter through the service
            snapshot = await self._capture_with_retry(context, dl_context)
            
            if snapshot:
                # Update capture stats
                capture_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                self._update_capture_stats(True, capture_time)
                
                # Store timing data
                context.timing_data['total_capture_ms'] = capture_time
                context.timing_data['datalayer_size_bytes'] = len(str(snapshot.latest_data))
                
                logger.debug(f"DataLayer capture completed for {context.page_url} in {capture_time:.1f}ms")
                return snapshot
            else:
                self._update_capture_stats(False, 0)
                return None
                
        except Exception as e:
            self._update_capture_stats(False, 0)
            logger.error(f"Coordinated dataLayer capture failed for {context.page_url}: {e}")
            
            if self.integration_config.fallback_on_capture_failure:
                return await self._fallback_capture(context)
            else:
                raise BrowserIntegrationError(f"DataLayer capture failed: {e}") from e
    
    async def _initialize_capture_session(self, context: CaptureContext) -> None:
        """Initialize capture session with browser preparation.
        
        Args:
            context: Capture context to initialize
        """
        if not context.page:
            raise BrowserIntegrationError("No page object in capture context")
        
        # Record page load timing
        context.page_load_timestamp = datetime.utcnow()
        
        # Wait for initial page stability
        await asyncio.sleep(self.integration_config.wait_after_load_ms / 1000)
        
        # Check for SPA navigation patterns
        await self._detect_spa_navigation(context)
        
        logger.debug(f"Initialized capture session for {context.page_url}")
    
    async def _detect_spa_navigation(self, context: CaptureContext) -> None:
        """Detect single-page application navigation patterns.
        
        Args:
            context: Capture context
        """
        if not self.integration_config.handle_spa_navigation:
            return
        
        try:
            # Check for common SPA frameworks and patterns
            spa_indicators = await context.page.evaluate("""
                () => {
                    const indicators = {
                        hasReact: !!(window.React || window.__REACT_DEVTOOLS_GLOBAL_HOOK__),
                        hasAngular: !!(window.angular || window.ng),
                        hasVue: !!(window.Vue),
                        hasHistory: !!(window.history && window.history.pushState),
                        hasSpaRouter: !!(window.location.hash || document.querySelector('[data-router]')),
                        hasAsyncContent: document.querySelectorAll('[data-async], [data-lazy]').length > 0
                    };
                    return indicators;
                }
            """)
            
            context.spa_navigation_detected = any(spa_indicators.values())
            context.metadata['spa_indicators'] = spa_indicators
            
            if context.spa_navigation_detected:
                logger.debug(f"SPA navigation detected for {context.page_url}")
                # Add additional wait time for SPA content loading
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.warning(f"SPA detection failed for {context.page_url}: {e}")
    
    async def _inject_monitoring_scripts(self, context: CaptureContext) -> None:
        """Inject dataLayer monitoring scripts into the page.
        
        Args:
            context: Capture context
        """
        try:
            monitoring_script = """
                (function() {
                    if (window.__dl_monitor_injected) return;
                    window.__dl_monitor_injected = true;
                    
                    // Monitor dataLayer changes
                    window.__dl_changes = [];
                    window.__dl_ready = false;
                    
                    // Check if dataLayer exists and is ready
                    function checkDataLayerReady() {
                        const dl = window.dataLayer || window.digitalData || window._satellite?.getVar?.('data layer');
                        if (dl) {
                            window.__dl_ready = true;
                            return true;
                        }
                        return false;
                    }
                    
                    // Monitor for dataLayer creation
                    let checkInterval = setInterval(() => {
                        if (checkDataLayerReady()) {
                            clearInterval(checkInterval);
                        }
                    }, 100);
                    
                    // Clear interval after timeout
                    setTimeout(() => clearInterval(checkInterval), 10000);
                    
                    // Initial check
                    checkDataLayerReady();
                })();
            """
            
            await context.page.evaluate(monitoring_script)
            logger.debug(f"Monitoring scripts injected for {context.page_url}")
            
        except Exception as e:
            logger.warning(f"Failed to inject monitoring scripts for {context.page_url}: {e}")
    
    async def _wait_for_datalayer_ready(self, context: CaptureContext) -> None:
        """Wait for dataLayer to be ready for capture.
        
        Args:
            context: Capture context
        """
        max_wait_ms = self.integration_config.max_wait_for_datalayer_ms
        check_interval_ms = 200
        elapsed_ms = 0
        
        while elapsed_ms < max_wait_ms:
            try:
                is_ready = await context.page.evaluate("""
                    () => {
                        // Check if monitoring script detected readiness
                        if (window.__dl_ready) return true;
                        
                        // Fallback: direct dataLayer check
                        return !!(window.dataLayer || window.digitalData || window._satellite);
                    }
                """)
                
                if is_ready:
                    context.datalayer_ready = True
                    logger.debug(f"DataLayer ready for {context.page_url} after {elapsed_ms}ms")
                    return
                
                await asyncio.sleep(check_interval_ms / 1000)
                elapsed_ms += check_interval_ms
                
            except Exception as e:
                logger.warning(f"Error checking dataLayer readiness for {context.page_url}: {e}")
                break
        
        # Timeout reached
        if self.integration_config.timeout_behavior == "fail":
            raise BrowserIntegrationError(f"DataLayer not ready after {max_wait_ms}ms")
        elif self.integration_config.timeout_behavior == "retry":
            # This will be handled by the retry logic in _capture_with_retry
            pass
        else:  # continue
            logger.warning(f"DataLayer not ready after {max_wait_ms}ms, continuing anyway")
            context.datalayer_ready = False
    
    async def _wait_for_capture_readiness(self, context: CaptureContext) -> None:
        """Wait for overall capture readiness including other captures.
        
        Args:
            context: Capture context
        """
        if not self.integration_config.coordinate_with_other_captures:
            return
        
        # Add small delay to allow other capture operations to initialize
        await asyncio.sleep(self.integration_config.capture_delay_ms / 1000)
        
        # In a real implementation, this would coordinate with other capture systems
        # For now, we'll simulate coordination by checking page stability
        try:
            # Wait for network idle (no requests for 500ms)
            await context.page.wait_for_load_state('networkidle', timeout=5000)
            context.other_captures_completed = True
        except Exception as e:
            logger.debug(f"Network idle timeout for {context.page_url}: {e}")
            context.other_captures_completed = False
    
    async def _capture_with_retry(
        self,
        context: CaptureContext,
        dl_context: DLContext
    ) -> Optional[DataLayerSnapshot]:
        """Capture dataLayer with retry logic.
        
        Args:
            context: Capture context
            dl_context: DataLayer context for capture
            
        Returns:
            DataLayer snapshot or None if all retries failed
        """
        max_retries = self.integration_config.max_capture_retries
        
        for attempt in range(max_retries + 1):
            try:
                # Create snapshotter instance
                snapshotter = Snapshotter(self.datalayer_service.config.capture)
                
                # Perform capture using the page object
                snapshot = await snapshotter.capture_from_page(context.page, dl_context)
                
                if snapshot and (snapshot.latest_data or snapshot.events_data):
                    return snapshot
                else:
                    logger.warning(f"Empty snapshot captured for {context.page_url} (attempt {attempt + 1})")
                    if attempt < max_retries:
                        await asyncio.sleep(0.5 * (attempt + 1))  # Progressive backoff
                    
            except Exception as e:
                logger.warning(f"Capture attempt {attempt + 1} failed for {context.page_url}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))  # Progressive backoff
                else:
                    logger.error(f"All capture attempts failed for {context.page_url}")
        
        return None
    
    async def _fallback_capture(self, context: CaptureContext) -> Optional[DataLayerSnapshot]:
        """Perform fallback capture with minimal requirements.
        
        Args:
            context: Capture context
            
        Returns:
            Basic dataLayer snapshot or None
        """
        try:
            logger.info(f"Attempting fallback capture for {context.page_url}")
            
            # Simple capture without advanced features
            basic_data = await context.page.evaluate("""
                () => {
                    const data = {};
                    if (window.dataLayer) {
                        data.dataLayer = window.dataLayer.slice ? window.dataLayer.slice() : window.dataLayer;
                    }
                    if (window.digitalData) {
                        data.digitalData = window.digitalData;
                    }
                    return data;
                }
            """)
            
            if basic_data and (basic_data.get('dataLayer') or basic_data.get('digitalData')):
                # Create minimal snapshot
                snapshot = DataLayerSnapshot(
                    page_url=context.page_url,
                    timestamp=datetime.utcnow(),
                    latest_data=basic_data,
                    events_data=[],
                    capture_method="fallback",
                    success=True
                )
                
                logger.info(f"Fallback capture successful for {context.page_url}")
                return snapshot
            
            return None
            
        except Exception as e:
            logger.error(f"Fallback capture failed for {context.page_url}: {e}")
            return None
    
    async def _execute_hooks(self, hook_type: str, context: CaptureContext) -> None:
        """Execute registered hooks for the given type.
        
        Args:
            hook_type: Type of hooks to execute
            context: Capture context
        """
        for hook in self.capture_hooks.get(hook_type, []):
            try:
                await hook(context)
            except Exception as e:
                logger.error(f"Hook {hook.__name__} ({hook_type}) failed: {e}")
    
    def _update_capture_stats(self, success: bool, capture_time_ms: float) -> None:
        """Update capture performance statistics.
        
        Args:
            success: Whether the capture was successful
            capture_time_ms: Time taken for capture in milliseconds
        """
        self.capture_stats['total_captures'] += 1
        
        if success:
            self.capture_stats['successful_captures'] += 1
            # Update rolling average
            current_avg = self.capture_stats['avg_capture_time_ms']
            total_successful = self.capture_stats['successful_captures']
            self.capture_stats['avg_capture_time_ms'] = (
                (current_avg * (total_successful - 1) + capture_time_ms) / total_successful
            )
        else:
            self.capture_stats['failed_captures'] += 1
    
    def get_capture_statistics(self) -> Dict[str, Any]:
        """Get capture performance statistics.
        
        Returns:
            Dictionary with capture statistics
        """
        total = self.capture_stats['total_captures']
        if total == 0:
            return {
                'total_captures': 0,
                'success_rate': 0.0,
                'avg_capture_time_ms': 0.0,
                'active_sessions': 0
            }
        
        return {
            'total_captures': total,
            'successful_captures': self.capture_stats['successful_captures'],
            'failed_captures': self.capture_stats['failed_captures'],
            'success_rate': (self.capture_stats['successful_captures'] / total) * 100,
            'avg_capture_time_ms': round(self.capture_stats['avg_capture_time_ms'], 2),
            'active_sessions': len(self.active_captures)
        }
    
    def reset_statistics(self) -> None:
        """Reset capture performance statistics."""
        self.capture_stats = {
            'total_captures': 0,
            'successful_captures': 0,
            'failed_captures': 0,
            'avg_capture_time_ms': 0.0
        }
        logger.debug("Capture statistics reset")


class DataLayerCaptureManager:
    """High-level manager for coordinated dataLayer captures across multiple pages."""
    
    def __init__(
        self,
        datalayer_service: DataLayerService,
        integration_config: Optional[BrowserIntegrationConfig] = None
    ):
        """Initialize capture manager.
        
        Args:
            datalayer_service: DataLayer service instance
            integration_config: Browser integration configuration
        """
        self.datalayer_service = datalayer_service
        self.integrator = DataLayerBrowserIntegrator(datalayer_service, integration_config)
        
        # Batch processing
        self.batch_contexts: List[CaptureContext] = []
        self.batch_results: List[DataLayerSnapshot] = []
    
    async def capture_page(
        self,
        page: Any,  # Playwright Page
        page_url: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[DataLayerSnapshot]:
        """Capture dataLayer for a single page with full coordination.
        
        Args:
            page: Playwright Page object
            page_url: URL of the page
            metadata: Additional metadata
            
        Returns:
            DataLayer snapshot or None
        """
        async with self.integrator.capture_session(page, page_url, metadata) as context:
            return await self.integrator.capture_datalayer_coordinated(context)
    
    async def capture_multiple_pages(
        self,
        page_contexts: List[Dict[str, Any]]
    ) -> List[Optional[DataLayerSnapshot]]:
        """Capture dataLayer for multiple pages concurrently.
        
        Args:
            page_contexts: List of dicts with 'page', 'url', and optional 'metadata'
            
        Returns:
            List of DataLayer snapshots (may contain None values)
        """
        tasks = []
        
        for page_context in page_contexts:
            task = self.capture_page(
                page_context['page'],
                page_context['url'],
                page_context.get('metadata')
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        snapshots = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Page {page_contexts[i]['url']} capture failed: {result}")
                snapshots.append(None)
            else:
                snapshots.append(result)
        
        return snapshots
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get overall integration status and health.
        
        Returns:
            Integration status information
        """
        return {
            'playwright_available': HAS_PLAYWRIGHT,
            'active_captures': len(self.integrator.active_captures),
            'capture_statistics': self.integrator.get_capture_statistics(),
            'configuration': {
                'coordinate_with_other_captures': self.integrator.integration_config.coordinate_with_other_captures,
                'inject_capture_script': self.integrator.integration_config.inject_capture_script,
                'handle_spa_navigation': self.integrator.integration_config.handle_spa_navigation,
                'max_wait_for_datalayer_ms': self.integrator.integration_config.max_wait_for_datalayer_ms
            }
        }