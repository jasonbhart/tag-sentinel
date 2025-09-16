"""Network request observer for capturing complete request lifecycle.

This module provides the NetworkObserver class that hooks into Playwright
network events to capture complete request/response data with accurate
timing information, headers, and error handling.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Set

from playwright.async_api import Page, Request, Response

from ..models.capture import (
    RequestLog, 
    RequestStatus, 
    ResourceType, 
    TimingData
)

logger = logging.getLogger(__name__)


class NetworkObserver:
    """Observes network requests and builds complete RequestLog records."""
    
    def __init__(self, page: Page):
        """Initialize network observer for a page.

        Args:
            page: Playwright page to observe
        """
        self.page = page
        self.requests: Dict[str, RequestLog] = {}  # URL-keyed for compatibility
        self._requests_by_id: Dict[str, RequestLog] = {}  # ID-keyed for correctness
        self.completed_requests: List[RequestLog] = []
        self._active_requests: Set[str] = set()
        self._callbacks: List[Callable[[RequestLog], None]] = []

        # Track request start times for accurate timing (ID-keyed)
        self._request_start_times: Dict[str, datetime] = {}
        
        # Setup event listeners
        self._setup_listeners()
    
    def _setup_listeners(self) -> None:
        """Setup Playwright event listeners for network events."""
        self.page.on("request", self._on_request)
        self.page.on("response", self._on_response) 
        self.page.on("requestfinished", self._on_request_finished)
        self.page.on("requestfailed", self._on_request_failed)
        
        logger.debug("Network observer listeners setup complete")
    
    def add_callback(self, callback: Callable[[RequestLog], None]) -> None:
        """Add callback to be called when requests are completed.
        
        Args:
            callback: Function to call with completed RequestLog
        """
        self._callbacks.append(callback)
    
    def _create_request_log(self, request: Request, request_id: str) -> RequestLog:
        """Create initial RequestLog from Playwright Request.
        
        Args:
            request: Playwright request object
            
        Returns:
            Initial RequestLog with request data
        """
        # Map Playwright resource types to our enum
        resource_type_map = {
            'document': ResourceType.DOCUMENT,
            'stylesheet': ResourceType.STYLESHEET,
            'image': ResourceType.IMAGE,
            'media': ResourceType.MEDIA,
            'font': ResourceType.FONT,
            'script': ResourceType.SCRIPT,
            'texttrack': ResourceType.TEXTTRACK,
            'xhr': ResourceType.XHR,
            'fetch': ResourceType.FETCH,
            'eventsource': ResourceType.EVENTSOURCE,
            'websocket': ResourceType.WEBSOCKET,
            'manifest': ResourceType.MANIFEST,
            'other': ResourceType.OTHER,
        }
        
        resource_type = resource_type_map.get(
            request.resource_type, 
            ResourceType.OTHER
        )
        
        # Extract headers
        headers = {}
        try:
            headers = request.headers
        except Exception as e:
            logger.warning(f"Failed to extract request headers: {e}")
        
        # Extract request body if available
        request_body = None
        try:
            if request.method.upper() in ('POST', 'PUT', 'PATCH'):
                request_body = request.post_data
        except Exception as e:
            logger.debug(f"Failed to extract request body: {e}")
        
        return RequestLog(
            url=request.url,
            method=request.method,
            resource_type=resource_type,
            request_headers=headers,
            request_body=request_body,
            start_time=self._request_start_times.get(request_id, datetime.utcnow()),
            status=RequestStatus.PENDING
        )
    
    def _extract_timing_data(self, request: Request) -> Optional[TimingData]:
        """Extract timing data from Playwright request.
        
        Args:
            request: Playwright request object
            
        Returns:
            TimingData object or None if unavailable
        """
        try:
            # Get timing information from request
            timing = request.timing
            if not timing:
                return None
            
            return TimingData(
                dns_start=timing.get('domainLookupStart'),
                dns_end=timing.get('domainLookupEnd'),
                connect_start=timing.get('connectStart'),
                connect_end=timing.get('connectEnd'),
                request_start=timing.get('requestStart'),
                response_start=timing.get('responseStart'),
                response_end=timing.get('responseEnd'),
            )
            
        except Exception as e:
            logger.debug(f"Failed to extract timing data: {e}")
            return None
    
    def _on_request(self, request: Request) -> None:
        """Handle request start event.

        Args:
            request: Playwright request object
        """
        request_id = str(id(request))
        self._request_start_times[request_id] = datetime.utcnow()

        try:
            request_log = self._create_request_log(request, request_id)
            # Store by unique ID for correctness (prevents overwrites)
            self._requests_by_id[request_id] = request_log
            # Also store by URL for test compatibility (latest request wins)
            self.requests[request.url] = request_log
            self._active_requests.add(request_id)

            logger.debug(f"Request started: {request.method} {request.url}")

        except Exception as e:
            logger.error(f"Error processing request start: {e}")
    
    def _on_response(self, response: Response) -> None:
        """Handle response received event.
        
        Args:
            response: Playwright response object
        """
        # Use asyncio to handle the async parts in a synchronous context
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._process_response_async(response))
        except RuntimeError:
            # No event loop running, handle synchronously without async parts
            self._process_response_sync(response)
    
    async def _process_response_async(self, response: Response) -> None:
        """Process response asynchronously."""
        request = response.request
        request_id = str(id(request))

        if request_id not in self._requests_by_id:
            logger.warning(f"Response received for unknown request: {request_id}")
            return

        try:
            request_log = self._requests_by_id[request_id]
            
            # Update with response data
            request_log.status_code = response.status
            request_log.status_text = response.status_text
            
            # Extract response headers
            try:
                request_log.response_headers = response.headers
            except Exception as e:
                logger.warning(f"Failed to extract response headers: {e}")
            
            # Extract response body (if small and text-based)
            try:
                content_type = response.headers.get('content-type', '')
                content_length = int(response.headers.get('content-length', 0))
                
                # Only capture small text responses to avoid memory issues
                if (content_length > 0 and content_length < 50000 and
                    any(t in content_type.lower() for t in ['text', 'json', 'xml', 'javascript'])):
                    request_log.response_body = await response.text()
                    
            except Exception as e:
                logger.debug(f"Failed to extract response body: {e}")
            
            # Extract response size
            try:
                request_log.response_size = len(await response.body())
            except Exception as e:
                logger.debug(f"Failed to get response size: {e}")
            
            # Extract protocol information
            try:
                request_log.protocol = response.headers.get('http-version', 'HTTP/1.1')
                # Try to get remote address from response
                # Note: Playwright doesn't directly expose remote IP
                request_log.remote_address = None
            except Exception as e:
                logger.debug(f"Failed to extract protocol info: {e}")

            # Extract timing data
            request_log.timing = self._extract_timing_data(request)

            # Refresh URL mapping for test compatibility
            self.requests[request.url] = request_log

            logger.debug(f"Response received: {response.status} {request.url}")

        except Exception as e:
            logger.error(f"Error processing response: {e}")

    def _process_response_sync(self, response: Response) -> None:
        """Process response synchronously (without body/text content)."""
        request = response.request
        request_id = str(id(request))

        if request_id not in self._requests_by_id:
            logger.warning(f"Response received for unknown request: {request_id}")
            return

        try:
            request_log = self._requests_by_id[request_id]
            
            # Update with response data
            request_log.status_code = response.status
            request_log.status_text = response.status_text
            
            # Extract response headers
            try:
                request_log.response_headers = response.headers
            except Exception as e:
                logger.warning(f"Failed to extract response headers: {e}")
            
            # Skip response body and size extraction in sync mode
            
            # Extract protocol information
            try:
                request_log.protocol = response.headers.get('http-version', 'HTTP/1.1')
                request_log.remote_address = None
            except Exception as e:
                logger.debug(f"Failed to extract protocol info: {e}")
            
            # Extract timing data
            request_log.timing = self._extract_timing_data(request)

            # Refresh URL mapping for test compatibility
            self.requests[request.url] = request_log

            logger.debug(f"Response received: {response.status} {request.url}")

        except Exception as e:
            logger.error(f"Error processing response: {e}")

    def _on_request_finished(self, request: Request) -> None:
        """Handle request finished event.
        
        Args:
            request: Playwright request object
        """
        request_id = str(id(request))

        if request_id not in self._requests_by_id:
            logger.warning(f"Request finished for unknown request: {request_id}")
            return

        try:
            request_log = self._requests_by_id[request_id]
            request_log.status = RequestStatus.SUCCESS
            request_log.end_time = datetime.utcnow()
            
            # Move to completed list
            self.completed_requests.append(request_log)
            self._active_requests.discard(request_id)
            
            # Refresh URL mapping for test compatibility
            self.requests[request.url] = request_log

            # Call callbacks
            for callback in self._callbacks:
                try:
                    callback(request_log)
                except Exception as e:
                    logger.error(f"Error in request callback: {e}")

            logger.debug(f"Request finished: {request.method} {request.url}")
            
        except Exception as e:
            logger.error(f"Error processing request finish: {e}")
    
    def _on_request_failed(self, request: Request) -> None:
        """Handle request failed event.
        
        Args:
            request: Playwright request object
        """
        request_id = str(id(request))

        if request_id not in self._requests_by_id:
            logger.warning(f"Request failed for unknown request: {request_id}")
            # Create a failed request log for unknown requests
            request_log = self._create_request_log(request, request_id)
            self._requests_by_id[request_id] = request_log
            self.requests[request.url] = request_log

        try:
            request_log = self._requests_by_id[request_id]
            
            # Determine failure type
            error_text = request.failure or "Unknown error"
            
            if "timeout" in error_text.lower():
                request_log.status = RequestStatus.TIMEOUT
            elif "abort" in error_text.lower():
                request_log.status = RequestStatus.ABORTED
            else:
                request_log.status = RequestStatus.FAILED
            
            request_log.error_text = error_text
            request_log.end_time = datetime.utcnow()
            
            # Move to completed list
            self.completed_requests.append(request_log)
            self._active_requests.discard(request_id)

            # Refresh URL mapping for test compatibility
            self.requests[request.url] = request_log

            # Call callbacks
            for callback in self._callbacks:
                try:
                    callback(request_log)
                except Exception as e:
                    logger.error(f"Error in request callback: {e}")

            logger.debug(f"Request failed: {request.method} {request.url} - {error_text}")
            
        except Exception as e:
            logger.error(f"Error processing request failure: {e}")
    
    def get_completed_requests(self) -> List[RequestLog]:
        """Get all completed requests.
        
        Returns:
            List of completed RequestLog objects
        """
        return self.completed_requests.copy()
    
    def get_active_requests(self) -> List[RequestLog]:
        """Get currently active (pending) requests.

        Returns:
            List of active RequestLog objects
        """
        active = []
        for request_id in self._active_requests:
            if request_id in self._requests_by_id:
                active.append(self._requests_by_id[request_id])
        return active
    
    def get_all_requests(self) -> List[RequestLog]:
        """Get all requests (completed and active).
        
        Returns:
            List of all RequestLog objects
        """
        return self.completed_requests + self.get_active_requests()
    
    def get_requests_by_host(self, host: str) -> List[RequestLog]:
        """Get all requests for a specific host.
        
        Args:
            host: Hostname to filter by
            
        Returns:
            List of RequestLog objects for the host
        """
        return [req for req in self.get_all_requests() if req.host == host]
    
    def get_requests_by_type(self, resource_type: ResourceType) -> List[RequestLog]:
        """Get all requests of a specific resource type.
        
        Args:
            resource_type: Resource type to filter by
            
        Returns:
            List of RequestLog objects of the specified type
        """
        return [req for req in self.get_all_requests() if req.resource_type == resource_type]
    
    def get_failed_requests(self) -> List[RequestLog]:
        """Get all failed requests.
        
        Returns:
            List of failed RequestLog objects
        """
        return [req for req in self.get_all_requests() if not req.is_successful]
    
    def finalize_pending_requests(self) -> None:
        """Mark all pending requests as failed (for cleanup).
        
        This should be called when page navigation is complete to handle
        any requests that never finished.
        """
        current_time = datetime.utcnow()
        
        for request_id in list(self._active_requests):
            if request_id in self._requests_by_id:
                request_log = self._requests_by_id[request_id]
                request_log.status = RequestStatus.TIMEOUT
                request_log.error_text = "Request did not complete before navigation ended"
                request_log.end_time = current_time
                
                self.completed_requests.append(request_log)
                self._active_requests.discard(request_id)
                
                logger.debug(f"Finalized pending request: {request_log.url}")
    
    def clear(self) -> None:
        """Clear all request data."""
        self.requests.clear()
        self.completed_requests.clear()
        self._active_requests.clear()
        self._request_start_times.clear()
        self._callbacks.clear()
        
        logger.debug("Network observer cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """Get network request statistics.
        
        Returns:
            Dictionary with request statistics
        """
        all_requests = self.get_all_requests()
        
        return {
            'total_requests': len(all_requests),
            'successful_requests': len([r for r in all_requests if r.is_successful]),
            'failed_requests': len([r for r in all_requests if not r.is_successful]),
            'pending_requests': len(self._active_requests),
            'timeout_requests': len([r for r in all_requests if r.status == RequestStatus.TIMEOUT]),
            'aborted_requests': len([r for r in all_requests if r.status == RequestStatus.ABORTED]),
            'document_requests': len(self.get_requests_by_type(ResourceType.DOCUMENT)),
            'script_requests': len(self.get_requests_by_type(ResourceType.SCRIPT)),
            'stylesheet_requests': len(self.get_requests_by_type(ResourceType.STYLESHEET)),
            'image_requests': len(self.get_requests_by_type(ResourceType.IMAGE)),
            'xhr_requests': len(self.get_requests_by_type(ResourceType.XHR)),
        }
    
    def __repr__(self) -> str:
        """String representation of network observer."""
        stats = self.get_stats()
        return (
            f"NetworkObserver(total={stats['total_requests']}, "
            f"completed={stats['total_requests'] - stats['pending_requests']}, "
            f"pending={stats['pending_requests']}, "
            f"failed={stats['failed_requests']})"
        )