"""Console and page error observers for capturing browser events.

This module provides ConsoleObserver and PageErrorObserver classes that
capture console messages and JavaScript errors from browser pages for
debugging and analysis purposes.
"""

import logging
from typing import List, Callable, Set, Dict, Any
from datetime import datetime

from playwright.async_api import Page, ConsoleMessage

from ..models.capture import ConsoleLog, ConsoleLevel

logger = logging.getLogger(__name__)


class ConsoleObserver:
    """Observer for console messages from browser pages."""
    
    def __init__(self, page: Page, filter_noise: bool = True):
        """Initialize console observer for a page.
        
        Args:
            page: Playwright page to observe
            filter_noise: Whether to filter out noisy console messages
        """
        self.page = page
        self.filter_noise = filter_noise
        self.console_logs: List[ConsoleLog] = []
        self._callbacks: List[Callable[[ConsoleLog], None]] = []
        
        # Noise filtering patterns
        self._noise_patterns = {
            "chrome_extensions": [
                "extension",
                "chrome-extension://",
                "moz-extension://",
                "webkitURL is deprecated",
                "Non-standard event",
            ],
            "development_tools": [
                "React DevTools",
                "Vue DevTools",
                "[HMR]",
                "[WDS]",
                "webpack",
                "__webpack",
                "DevTools failed to load",
            ],
            "analytics_noise": [
                "Failed to load resource: the server responded with a status of",
                "favicon.ico - Failed to load resource",
                "Blocked attempt to show a 'beforeunload'",
                "Image from origin",
            ],
            "performance_observer": [
                "PerformanceObserver",
                "ResizeObserver loop limit exceeded",
                "IntersectionObserver",
            ]
        }
        
        # Setup event listener
        self._setup_listener()
    
    def _setup_listener(self) -> None:
        """Setup Playwright console event listener."""
        self.page.on("console", self._on_console_message)
        logger.debug("Console observer listener setup complete")
    
    def add_callback(self, callback: Callable[[ConsoleLog], None]) -> None:
        """Add callback to be called when console messages are captured.
        
        Args:
            callback: Function to call with ConsoleLog
        """
        self._callbacks.append(callback)
    
    def _should_filter_message(self, message: str) -> bool:
        """Check if console message should be filtered out as noise.
        
        Args:
            message: Console message text
            
        Returns:
            True if message should be filtered out
        """
        if not self.filter_noise:
            return False
        
        message_lower = message.lower()
        
        # Check against noise patterns
        for category, patterns in self._noise_patterns.items():
            for pattern in patterns:
                if pattern.lower() in message_lower:
                    logger.debug(f"Filtering noise message ({category}): {message[:100]}")
                    return True
        
        # Filter out very long messages (likely data dumps)
        if len(message) > 5000:
            logger.debug(f"Filtering long message: {len(message)} chars")
            return True
        
        return False
    
    def _on_console_message(self, message: ConsoleMessage) -> None:
        """Handle console message event.
        
        Args:
            message: Playwright console message
        """
        try:
            # Create ConsoleLog from Playwright message
            console_log = ConsoleLog.from_playwright_message(message)
            
            # Filter noise if enabled
            if self._should_filter_message(console_log.text):
                return
            
            # Store console log
            self.console_logs.append(console_log)
            
            # Call callbacks
            for callback in self._callbacks:
                try:
                    callback(console_log)
                except Exception as e:
                    logger.error(f"Error in console callback: {e}")
            
            logger.debug(f"Console {console_log.level}: {console_log.text[:100]}")
            
        except Exception as e:
            logger.error(f"Error processing console message: {e}")
    
    def get_console_logs(self) -> List[ConsoleLog]:
        """Get all captured console logs.
        
        Returns:
            List of ConsoleLog objects
        """
        return self.console_logs.copy()
    
    def get_logs_by_level(self, level: ConsoleLevel) -> List[ConsoleLog]:
        """Get console logs filtered by level.
        
        Args:
            level: Console level to filter by
            
        Returns:
            List of ConsoleLog objects at specified level
        """
        return [log for log in self.console_logs if log.level == level]
    
    def get_error_logs(self) -> List[ConsoleLog]:
        """Get only error level console logs.
        
        Returns:
            List of error ConsoleLog objects
        """
        return self.get_logs_by_level(ConsoleLevel.ERROR)
    
    def get_warning_logs(self) -> List[ConsoleLog]:
        """Get only warning level console logs.
        
        Returns:
            List of warning ConsoleLog objects
        """
        return self.get_logs_by_level(ConsoleLevel.WARN)
    
    def clear(self) -> None:
        """Clear all console logs."""
        self.console_logs.clear()
        logger.debug("Console observer cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """Get console log statistics.
        
        Returns:
            Dictionary with console log statistics
        """
        return {
            'total_messages': len(self.console_logs),
            'error_messages': len(self.get_error_logs()),
            'warning_messages': len(self.get_warning_logs()),
            'info_messages': len(self.get_logs_by_level(ConsoleLevel.INFO)),
            'log_messages': len(self.get_logs_by_level(ConsoleLevel.LOG)),
            'debug_messages': len(self.get_logs_by_level(ConsoleLevel.DEBUG)),
        }
    
    def __repr__(self) -> str:
        """String representation of console observer."""
        stats = self.get_stats()
        return (
            f"ConsoleObserver(total={stats['total_messages']}, "
            f"errors={stats['error_messages']}, "
            f"warnings={stats['warning_messages']})"
        )


class PageErrorObserver:
    """Observer for JavaScript page errors and uncaught exceptions."""
    
    def __init__(self, page: Page):
        """Initialize page error observer.
        
        Args:
            page: Playwright page to observe
        """
        self.page = page
        self.page_errors: List[str] = []
        self._callbacks: List[Callable[[str], None]] = []
        
        # Setup event listener
        self._setup_listener()
    
    def _setup_listener(self) -> None:
        """Setup Playwright page error event listener."""
        self.page.on("pageerror", self._on_page_error)
        logger.debug("Page error observer listener setup complete")
    
    def add_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback to be called when page errors are captured.
        
        Args:
            callback: Function to call with error message
        """
        self._callbacks.append(callback)
    
    def _on_page_error(self, error: Exception) -> None:
        """Handle page error event.
        
        Args:
            error: JavaScript error exception
        """
        try:
            error_text = str(error)
            
            # Store error
            self.page_errors.append(error_text)
            
            # Call callbacks
            for callback in self._callbacks:
                try:
                    callback(error_text)
                except Exception as e:
                    logger.error(f"Error in page error callback: {e}")
            
            logger.warning(f"Page error: {error_text}")
            
        except Exception as e:
            logger.error(f"Error processing page error: {e}")
    
    def get_page_errors(self) -> List[str]:
        """Get all captured page errors.
        
        Returns:
            List of error message strings
        """
        return self.page_errors.copy()
    
    def has_errors(self) -> bool:
        """Check if any page errors were captured.
        
        Returns:
            True if page errors were captured
        """
        return len(self.page_errors) > 0
    
    def clear(self) -> None:
        """Clear all page errors."""
        self.page_errors.clear()
        logger.debug("Page error observer cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """Get page error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        return {
            'total_errors': len(self.page_errors),
        }
    
    def __repr__(self) -> str:
        """String representation of page error observer."""
        return f"PageErrorObserver(errors={len(self.page_errors)})"


class CombinedObserver:
    """Combined observer for both console messages and page errors."""
    
    def __init__(self, page: Page, filter_console_noise: bool = True):
        """Initialize combined observer.
        
        Args:
            page: Playwright page to observe
            filter_console_noise: Whether to filter console noise
        """
        self.page = page
        self.console_observer = ConsoleObserver(page, filter_console_noise)
        self.error_observer = PageErrorObserver(page)
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Setup cross-observer callbacks
        self.console_observer.add_callback(self._on_console_message)
        self.error_observer.add_callback(self._on_page_error)
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for any captured event.
        
        Args:
            callback: Function to call with event data dict
        """
        self._callbacks.append(callback)
    
    def _on_console_message(self, console_log: ConsoleLog) -> None:
        """Handle console message from ConsoleObserver.
        
        Args:
            console_log: Captured console log
        """
        event_data = {
            'type': 'console',
            'data': console_log,
            'timestamp': console_log.timestamp,
            'level': console_log.level.value,
        }
        
        for callback in self._callbacks:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error in combined observer console callback: {e}")
    
    def _on_page_error(self, error_text: str) -> None:
        """Handle page error from PageErrorObserver.
        
        Args:
            error_text: Error message text
        """
        event_data = {
            'type': 'page_error',
            'data': error_text,
            'timestamp': datetime.utcnow(),
            'level': 'error',
        }
        
        for callback in self._callbacks:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error in combined observer error callback: {e}")
    
    def get_all_events(self) -> List[Dict[str, Any]]:
        """Get all captured events in chronological order.
        
        Returns:
            List of event dictionaries with type, data, timestamp
        """
        events = []
        
        # Add console logs
        for log in self.console_observer.get_console_logs():
            events.append({
                'type': 'console',
                'data': log,
                'timestamp': log.timestamp,
                'level': log.level.value,
            })
        
        # Add page errors (with estimated timestamps)
        for i, error in enumerate(self.error_observer.get_page_errors()):
            # Estimate timestamp based on order (not perfect but better than nothing)
            base_time = datetime.utcnow()
            estimated_time = base_time.replace(microsecond=i * 1000)  # Spread errors by microseconds
            
            events.append({
                'type': 'page_error',
                'data': error,
                'timestamp': estimated_time,
                'level': 'error',
            })
        
        # Sort by timestamp
        events.sort(key=lambda x: x['timestamp'])
        return events
    
    def get_console_logs(self) -> List[ConsoleLog]:
        """Get console logs from console observer."""
        return self.console_observer.get_console_logs()
    
    def get_page_errors(self) -> List[str]:
        """Get page errors from error observer."""
        return self.error_observer.get_page_errors()
    
    def get_error_count(self) -> int:
        """Get total error count (console errors + page errors).
        
        Returns:
            Total number of errors
        """
        console_errors = len(self.console_observer.get_error_logs())
        page_errors = len(self.error_observer.get_page_errors())
        return console_errors + page_errors
    
    def clear(self) -> None:
        """Clear all captured data."""
        self.console_observer.clear()
        self.error_observer.clear()
        logger.debug("Combined observer cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """Get combined statistics.
        
        Returns:
            Dictionary with combined statistics
        """
        console_stats = self.console_observer.get_stats()
        error_stats = self.error_observer.get_stats()
        
        return {
            **console_stats,
            **error_stats,
            'total_errors': self.get_error_count(),
        }
    
    def __repr__(self) -> str:
        """String representation of combined observer."""
        stats = self.get_stats()
        return (
            f"CombinedObserver(console_messages={stats['total_messages']}, "
            f"page_errors={stats['total_errors']}, "
            f"total_errors={stats['total_errors']})"
        )