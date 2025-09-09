"""GA4 (Google Analytics 4) detector for network pattern matching and parameter extraction."""

import json
import httpx
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from .base import (
    BaseDetector, 
    DetectContext, 
    DetectResult, 
    TagEvent, 
    TagStatus, 
    Confidence, 
    Vendor,
    NoteCategory
)
from .errors import (
    ResilientDetector,
    ErrorSeverity,
    ErrorCategory, 
    resilient_operation,
    safe_request_processing,
    with_timeout
)
from .performance import (
    monitor_performance,
    optimized_pattern_match,
    filter_requests_for_processing,
    parameter_cache,
    get_performance_summary
)
from .utils import (
    patterns,
    ParameterParser,
    extract_measurement_id,
    extract_url_components,
    match_url_pattern,
    normalize_parameter_name,
    validate_measurement_id
)
from ..models.capture import PageResult, RequestLog


class GA4Detector(BaseDetector, ResilientDetector):
    """Detector for Google Analytics 4 network requests and events."""
    
    def __init__(self):
        super().__init__("GA4Detector", "1.0.0")
        self.parser = ParameterParser()
    
    @property
    def supported_vendors(self) -> Set[Vendor]:
        """GA4 detector supports GA4 vendor."""
        return {Vendor.GA4}
    
    async def detect(self, page: PageResult, ctx: DetectContext) -> DetectResult:
        """Analyze page for GA4 activity.
        
        Args:
            page: Page capture result with network requests
            ctx: Detection context with configuration
            
        Returns:
            Detection results with GA4 events and notes
        """
        result = self._create_result()
        start_time = datetime.utcnow()
        
        self.error_collector = self._create_error_collector()
        
        with self.error_context("ga4_detection", page_url=page.url):
            # Process network requests for GA4 patterns
            ga4_requests = self._find_ga4_requests_safe(page.network_requests)
            result.processed_requests = len(ga4_requests)
            
            # Extract events from each GA4 request with error handling
            self._extract_events_safe(ga4_requests, page.url, ctx, result)
            
            # Add analysis notes
            with self.error_context("analysis_notes", page_url=page.url):
                self._add_analysis_notes(result, page.url, ga4_requests)
            
            # Run MP debug validation if enabled and not in production
            if (not ctx.is_production and 
                ctx.config.get("ga4", {}).get("mp_debug", {}).get("enabled", False)):
                with self.error_context("mp_debug_validation", page_url=page.url):
                    await self._validate_with_mp_debug_safe(result, ga4_requests, ctx)
            
            # Set page URL for all notes
            self._set_note_page_urls(result, page.url)
        
        # Add collected errors to result
        self.error_collector.add_to_result(result)
        
        # Set result success based on error severity
        if self.error_collector.has_critical_errors():
            result.success = False
            result.error_message = "Critical errors encountered during GA4 detection"
        elif self.error_collector.has_blocking_errors():
            result.success = False
            result.error_message = "Blocking errors prevented complete GA4 detection"
        
        # Calculate processing time
        end_time = datetime.utcnow()
        result.processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Add performance summary as metadata if available
        if hasattr(self, 'error_collector') and self.error_collector:
            perf_summary = get_performance_summary()
            if perf_summary.get('operations'):
                result.metrics.update({
                    'performance_summary': perf_summary,
                    'detector_operations': list(perf_summary.get('operations', {}).keys())
                })
        
        return result
    
    @monitor_performance("find_ga4_requests")  
    @resilient_operation("find_ga4_requests", ErrorSeverity.LOW, ErrorCategory.PARSING, [])
    def _find_ga4_requests_safe(self, requests: List[RequestLog]) -> List[RequestLog]:
        """Safely find GA4 requests with error handling and performance optimization."""
        # First, filter requests to reduce processing overhead
        filtered_requests = filter_requests_for_processing(requests)
        return self._find_ga4_requests(filtered_requests)
    
    def _find_ga4_requests(self, requests: List[RequestLog]) -> List[RequestLog]:
        """Find all requests that match GA4 patterns.
        
        Args:
            requests: List of network requests to analyze
            
        Returns:
            List of requests that match GA4 endpoints
        """
        ga4_requests = []
        
        for request in requests:
            if self._is_ga4_request(request.url):
                ga4_requests.append(request)
        
        return ga4_requests
    
    @monitor_performance("ga4_pattern_match")
    def _is_ga4_request(self, url: str) -> bool:
        """Check if URL matches any GA4 endpoint patterns using optimized matching.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL matches GA4 patterns
        """
        # Use optimized pattern matching with caching
        ga4_patterns = [
            r"https://www\.google-analytics\.com/mp/collect",
            r"https://region1\.google-analytics\.com/mp/collect", 
            r"https://.*\.google-analytics\.com/mp/collect",
            r"https://www\.google-analytics\.com/g/collect"
        ]
        
        for pattern in ga4_patterns:
            if optimized_pattern_match(url, pattern):
                return True
        
        return False
    
    def _extract_events_safe(self, requests: List[RequestLog], page_url: str, 
                           ctx: DetectContext, result: DetectResult) -> None:
        """Safely extract events from GA4 requests with comprehensive error handling."""
        def process_request(request):
            return self._extract_events_from_request(request, page_url, ctx)
        
        # Use safe processing with error collection
        all_events = safe_request_processing(
            requests, 
            process_request,
            max_failures=len(requests) // 2,  # Stop if more than half fail
            error_collector=self.error_collector
        )
        
        # Add all successfully processed events
        for events in all_events:
            if events:
                for event in events:
                    result.add_event(event)
    
    def _extract_events_from_request(self, request: RequestLog, 
                                   page_url: str, ctx: DetectContext) -> List[TagEvent]:
        """Extract GA4 events from a network request.
        
        Args:
            request: Network request to analyze
            page_url: URL of the page being analyzed
            ctx: Detection context
            
        Returns:
            List of extracted TagEvent objects
        """
        events = []
        
        try:
            # Parse parameters from URL and body
            url_params = self._parse_url_parameters(request.url)
            body_params = self._parse_body_parameters(request)
            
            # Combine parameters, giving precedence to body params
            all_params = {**url_params, **body_params}
            
            # Extract measurement ID
            measurement_id = extract_measurement_id(request.url, all_params)
            
            # Extract GA4 events from parameters
            ga4_events = self.parser.extract_ga4_events(all_params)
            
            # Create TagEvent for each detected event
            for ga4_event in ga4_events:
                event = self._create_tag_event(
                    ga4_event,
                    request,
                    page_url,
                    measurement_id,
                    all_params,
                    ctx
                )
                events.append(event)
            
            # If no specific events found, create a generic GA4 hit event
            if not ga4_events and measurement_id:
                event = self._create_generic_ga4_event(
                    request, 
                    page_url, 
                    measurement_id, 
                    all_params,
                    ctx
                )
                events.append(event)
        
        except Exception as e:
            # Create error event for debugging
            error_event = TagEvent(
                vendor=Vendor.GA4,
                name="parsing_error",
                category="error",
                id=None,
                page_url=page_url,
                request_url=request.url,
                timing_ms=self._calculate_timing(request),
                status=TagStatus.ERROR,
                confidence=Confidence.LOW,
                params={"error": str(e)},
                detection_method="error_handling",
                detector_version=self.version
            )
            events.append(error_event)
        
        return events
    
    def _parse_url_parameters(self, url: str) -> Dict[str, Any]:
        """Parse parameters from URL query string.
        
        Args:
            url: URL to parse
            
        Returns:
            Dictionary of parsed parameters
        """
        components = extract_url_components(url)
        if components["query"]:
            return self.parser.parse_url_encoded(components["query"])
        return {}
    
    def _parse_body_parameters(self, request: RequestLog) -> Dict[str, Any]:
        """Parse parameters from request body.
        
        Args:
            request: Request with potential body data
            
        Returns:
            Dictionary of parsed parameters
        """
        if not request.request_body:
            return {}
        
        content_type = request.request_headers.get("content-type", "")
        return self.parser.parse_form_data(request.request_body, content_type)
    
    def _create_tag_event(self, ga4_event: Dict[str, Any], request: RequestLog,
                         page_url: str, measurement_id: Optional[str],
                         all_params: Dict[str, Any], ctx: DetectContext) -> TagEvent:
        """Create a TagEvent from GA4 event data.
        
        Args:
            ga4_event: Extracted GA4 event dict
            request: Original network request
            page_url: Page URL
            measurement_id: GA4 measurement ID
            all_params: All parsed parameters
            ctx: Detection context
            
        Returns:
            TagEvent object
        """
        # Determine event status based on request success
        status = TagStatus.OK if request.is_successful else TagStatus.ERROR
        
        # Determine confidence based on measurement ID presence and validation
        confidence = self._determine_confidence(measurement_id, ga4_event["name"])
        
        # Extract comprehensive parameter information
        enriched_params = self._enrich_event_parameters(
            ga4_event, all_params, request, measurement_id
        )
        
        return TagEvent(
            vendor=Vendor.GA4,
            name=ga4_event["name"],
            category="analytics_event",
            id=measurement_id,
            page_url=page_url,
            request_url=request.url,
            timing_ms=self._calculate_timing(request),
            status=status,
            confidence=confidence,
            params=enriched_params,
            detection_method="network_request",
            detector_version=self.version
        )
    
    def _enrich_event_parameters(self, ga4_event: Dict[str, Any], 
                               all_params: Dict[str, Any], request: RequestLog,
                               measurement_id: Optional[str]) -> Dict[str, Any]:
        """Enrich event parameters with comprehensive GA4 data extraction.
        
        Args:
            ga4_event: Basic GA4 event data
            all_params: All parsed parameters
            request: Network request
            measurement_id: GA4 measurement ID
            
        Returns:
            Enriched parameters dictionary
        """
        # Start with base event information
        enriched = {
            "event_name": ga4_event["name"],
            "measurement_id": measurement_id,
            "request_url": request.url,
            "request_method": request.method,
            "status_code": request.status_code,
            "endpoint_type": self._classify_endpoint(request.url)
        }
        
        # Extract client and session information
        client_info = self._extract_enhanced_client_info(all_params)
        enriched.update(client_info)
        
        # Extract page and navigation info
        page_info = self._extract_enhanced_page_info(all_params)
        enriched.update(page_info)
        
        # Extract e-commerce data if present
        ecommerce_info = self._extract_ecommerce_parameters(all_params)
        if ecommerce_info:
            enriched["ecommerce"] = ecommerce_info
        
        # Extract custom dimensions and metrics
        custom_data = self.parser.extract_custom_dimensions(all_params)
        enriched.update(custom_data)
        
        # Extract GA4-specific parameters
        ga4_specific = self._extract_ga4_specific_params(all_params)
        enriched.update(ga4_specific)
        
        # Include event-specific parameters
        event_params = ga4_event.get("params", {})
        custom_params = ga4_event.get("custom_params", {})
        
        # Merge event params with normalization
        for key, value in event_params.items():
            normalized_key = f"event_{normalize_parameter_name(key)}"
            enriched[normalized_key] = value
        
        for key, value in custom_params.items():
            normalized_key = f"custom_{normalize_parameter_name(key)}"
            enriched[normalized_key] = value
        
        # Add debugging information if enabled
        if all_params.get("_debug") == "1":
            enriched["debug_mode"] = True
        
        # Remove None values and empty strings
        enriched = {k: v for k, v in enriched.items() 
                   if v is not None and v != ""}
        
        return enriched
    
    def _extract_enhanced_client_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract enhanced client identification and device information.
        
        Args:
            params: Parsed parameters
            
        Returns:
            Dict with client information
        """
        client_info = self.parser.extract_client_info(params)
        
        # Add additional client-related parameters
        additional_client_params = {
            # Device and browser info
            "user_agent": params.get("ua"),
            "viewport_size": params.get("vp"),
            "screen_resolution": params.get("sr"), 
            "color_depth": params.get("sd"),
            "language": params.get("ul"),
            "encoding": params.get("de"),
            "timezone_offset": params.get("tmo"),
            
            # Client capabilities
            "java_enabled": params.get("je"),
            "flash_version": params.get("fl"),
            
            # App info (for mobile apps)
            "app_name": params.get("an"),
            "app_id": params.get("aid"), 
            "app_version": params.get("av"),
            "app_installer_id": params.get("aiid"),
            
            # Platform info
            "platform": params.get("p"),
            "platform_version": params.get("pv")
        }
        
        # Only include non-None values
        for key, value in additional_client_params.items():
            if value is not None:
                client_info[key] = value
        
        return client_info
    
    def _extract_enhanced_page_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract enhanced page and navigation information.
        
        Args:
            params: Parsed parameters
            
        Returns:
            Dict with page information
        """
        page_info = self.parser.extract_page_info(params)
        
        # Add additional page-related parameters
        additional_page_params = {
            # Navigation timing
            "page_load_time": params.get("plt"),
            "dns_time": params.get("dns"),
            "page_download_time": params.get("pdt"),
            "redirect_time": params.get("rrt"),
            "tcp_connect_time": params.get("tcp"),
            "server_response_time": params.get("srt"),
            "dom_interactive_time": params.get("dit"),
            "content_load_time": params.get("clt"),
            
            # Content info  
            "content_group_1": params.get("cg1"),
            "content_group_2": params.get("cg2"), 
            "content_group_3": params.get("cg3"),
            "content_group_4": params.get("cg4"),
            "content_group_5": params.get("cg5"),
            
            # Social interactions
            "social_network": params.get("sn"),
            "social_action": params.get("sa"),
            "social_target": params.get("st")
        }
        
        # Only include non-None values
        for key, value in additional_page_params.items():
            if value is not None:
                page_info[key] = value
        
        return page_info
    
    @resilient_operation("extract_ecommerce", ErrorSeverity.LOW, ErrorCategory.PARSING, None)
    def _extract_ecommerce_parameters(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract e-commerce related parameters.
        
        Args:
            params: Parsed parameters
            
        Returns:
            Dict with e-commerce data or None if not present
        """
        ecommerce_params = {}
        
        # Transaction-level parameters
        transaction_params = {
            "transaction_id": params.get("ti", params.get("tid")),
            "affiliation": params.get("ta"),
            "revenue": params.get("tr"),
            "tax": params.get("tt"),
            "shipping": params.get("ts"),
            "coupon": params.get("tcc"),
            "currency": params.get("cu")
        }
        
        # Item-level parameters (GA4 items array)
        if "items" in params and isinstance(params["items"], list):
            ecommerce_params["items"] = params["items"]
        
        # Enhanced e-commerce actions
        if "pa" in params:  # Product action
            ecommerce_params["product_action"] = params["pa"]
        
        if "pal" in params:  # Product action list
            ecommerce_params["product_action_list"] = params["pal"]
        
        # Include transaction params if any are present
        transaction_data = {k: v for k, v in transaction_params.items() if v is not None}
        if transaction_data:
            ecommerce_params.update(transaction_data)
        
        # Return None if no e-commerce data found
        return ecommerce_params if ecommerce_params else None
    
    def _extract_ga4_specific_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract GA4-specific parameters and settings.
        
        Args:
            params: Parsed parameters
            
        Returns:
            Dict with GA4-specific information
        """
        ga4_params = {}
        
        # GA4 Measurement Protocol specific
        if "firebase_app_id" in params:
            ga4_params["firebase_app_id"] = params["firebase_app_id"]
        
        if "app_instance_id" in params:
            ga4_params["app_instance_id"] = params["app_instance_id"]
        
        # Engagement parameters
        if "engagement_time_msec" in params:
            ga4_params["engagement_time_msec"] = params["engagement_time_msec"]
        
        if "session_engaged" in params:
            ga4_params["session_engaged"] = params["session_engaged"]
        
        # Session parameters
        if "session_number" in params:
            ga4_params["session_number"] = params["session_number"]
        
        if "_ss" in params:  # Session start
            ga4_params["session_start"] = params["_ss"] == "1"
        
        if "_fv" in params:  # First visit
            ga4_params["first_visit"] = params["_fv"] == "1"
        
        # Consent and privacy
        if "gcs" in params:  # Google Consent State
            ga4_params["consent_state"] = params["gcs"]
        
        if "dma" in params:  # Data Marketing Attribution
            ga4_params["dma"] = params["dma"]
        
        # Geographic data
        if "geoid" in params:
            ga4_params["geo_id"] = params["geoid"]
        
        return ga4_params
    
    def _create_generic_ga4_event(self, request: RequestLog, page_url: str,
                                 measurement_id: str, all_params: Dict[str, Any],
                                 ctx: DetectContext) -> TagEvent:
        """Create a generic GA4 event when specific events can't be identified.
        
        Args:
            request: Network request
            page_url: Page URL  
            measurement_id: GA4 measurement ID
            all_params: All parsed parameters
            ctx: Detection context
            
        Returns:
            Generic TagEvent
        """
        status = TagStatus.OK if request.is_successful else TagStatus.ERROR
        confidence = Confidence.MEDIUM if validate_measurement_id(measurement_id) else Confidence.LOW
        
        # Try to determine event type from URL patterns
        event_name = "page_view"  # Default assumption
        if "mp/collect" in request.url:
            event_name = "measurement_protocol"
        elif "g/collect" in request.url:
            event_name = "gtag_event"
        
        # Extract context information
        client_info = self.parser.extract_client_info(all_params)
        page_info = self.parser.extract_page_info(all_params)
        
        event_params = {
            "event_type": "generic",
            "measurement_id": measurement_id,
            "request_url": request.url,
            "request_method": request.method,
            "status_code": request.status_code,
            "endpoint_type": self._classify_endpoint(request.url),
            **client_info,
            **page_info
        }
        
        # Include a sample of other parameters for debugging
        other_params = {k: v for k, v in all_params.items() 
                       if k not in event_params and len(str(v)) < 100}
        if other_params:
            event_params["other_params"] = other_params
        
        return TagEvent(
            vendor=Vendor.GA4,
            name=event_name,
            category="generic",
            id=measurement_id,
            page_url=page_url,
            request_url=request.url,
            timing_ms=self._calculate_timing(request),
            status=status,
            confidence=confidence,
            params=event_params,
            detection_method="generic_pattern_match",
            detector_version=self.version
        )
    
    def _determine_confidence(self, measurement_id: Optional[str], event_name: str) -> Confidence:
        """Determine confidence level for event detection.
        
        Args:
            measurement_id: GA4 measurement ID if found
            event_name: Name of detected event
            
        Returns:
            Confidence level
        """
        # High confidence: Valid measurement ID and known event name
        if measurement_id and validate_measurement_id(measurement_id):
            known_events = {
                "page_view", "scroll", "click", "file_download", "video_start",
                "video_progress", "video_complete", "purchase", "add_to_cart",
                "begin_checkout", "login", "sign_up", "search"
            }
            if event_name in known_events:
                return Confidence.HIGH
            return Confidence.MEDIUM
        
        # Medium confidence: Has measurement ID but format might be off
        if measurement_id:
            return Confidence.MEDIUM
        
        # Low confidence: No measurement ID found
        return Confidence.LOW
    
    def _classify_endpoint(self, url: str) -> str:
        """Classify the GA4 endpoint type.
        
        Args:
            url: Request URL
            
        Returns:
            Endpoint classification
        """
        if match_url_pattern(url, "ga4_mp_collect"):
            return "measurement_protocol"
        elif match_url_pattern(url, "ga4_g_collect"):
            return "gtag_collect"
        elif match_url_pattern(url, "ga4_regional_endpoint"):
            return "regional_mp"
        else:
            return "unknown"
    
    def _calculate_timing(self, request: RequestLog) -> Optional[int]:
        """Calculate request timing in milliseconds.
        
        Args:
            request: Network request
            
        Returns:
            Timing in milliseconds or None
        """
        if request.timing and request.timing.total_time:
            return int(request.timing.total_time)
        elif request.duration_ms:
            return int(request.duration_ms)
        return None
    
    def _add_analysis_notes(self, result: DetectResult, page_url: str, 
                           ga4_requests: List[RequestLog]) -> None:
        """Add analysis notes based on detected GA4 activity.
        
        Args:
            result: Detection result to add notes to
            page_url: Page URL
            ga4_requests: List of GA4 requests found
        """
        if not ga4_requests:
            result.add_info_note(
                "No GA4 requests detected on this page",
                category=NoteCategory.DATA_QUALITY
            )
            return
        
        # Note about number of requests
        if len(ga4_requests) > 10:
            result.add_warning_note(
                f"Large number of GA4 requests detected ({len(ga4_requests)}). This may indicate duplicate events or excessive tracking.",
                category=NoteCategory.PERFORMANCE,
                related_events=[],
                request_count=len(ga4_requests)
            )
        
        # Check for measurement ID consistency
        measurement_ids = set()
        for request in ga4_requests:
            url_params = self._parse_url_parameters(request.url)
            body_params = self._parse_body_parameters(request)
            all_params = {**url_params, **body_params}
            
            mid = extract_measurement_id(request.url, all_params)
            if mid:
                measurement_ids.add(mid)
        
        if len(measurement_ids) > 1:
            result.add_warning_note(
                f"Multiple GA4 measurement IDs detected on same page: {list(measurement_ids)}",
                category=NoteCategory.CONFIGURATION,
                measurement_ids=list(measurement_ids)
            )
        elif len(measurement_ids) == 1:
            mid = list(measurement_ids)[0]
            if not validate_measurement_id(mid):
                result.add_warning_note(
                    f"GA4 measurement ID format appears invalid: {mid}",
                    category=NoteCategory.VALIDATION,
                    measurement_id=mid
                )
        
        # Check for failed requests
        failed_requests = [req for req in ga4_requests if not req.is_successful]
        if failed_requests:
            result.add_error_note(
                f"{len(failed_requests)} GA4 requests failed",
                category=NoteCategory.VALIDATION,
                failed_count=len(failed_requests),
                failed_urls=[req.url for req in failed_requests]
            )
        
        # Performance note for slow requests
        slow_requests = []
        for request in ga4_requests:
            timing = self._calculate_timing(request)
            if timing and timing > 5000:  # 5 second threshold
                slow_requests.append(request.url)
        
        if slow_requests:
            result.add_warning_note(
                f"{len(slow_requests)} GA4 requests took longer than 5 seconds",
                category=NoteCategory.PERFORMANCE,
                slow_requests=slow_requests
            )
    
    async def _validate_with_mp_debug_safe(self, result: DetectResult, 
                                          ga4_requests: List[RequestLog], 
                                          ctx: DetectContext) -> None:
        """Safely validate GA4 requests using MP debug endpoint with error handling."""
        try:
            await self._validate_with_mp_debug(result, ga4_requests, ctx)
        except Exception as e:
            # Add error but don't break execution
            self.error_collector.add_from_exception(
                e, "mp_debug_validation", 
                ErrorSeverity.MEDIUM, 
                ErrorCategory.EXTERNAL_API,
                request_count=len(ga4_requests)
            )
    
    async def _validate_with_mp_debug(self, result: DetectResult, 
                                     ga4_requests: List[RequestLog], 
                                     ctx: DetectContext) -> None:
        """Validate GA4 requests using Measurement Protocol debug endpoint.
        
        Args:
            result: Detection result to add validation notes to
            ga4_requests: List of GA4 requests to validate
            ctx: Detection context
        """
        if ctx.is_production:
            result.add_warning_note(
                "MP debug validation skipped in production environment",
                category=NoteCategory.VALIDATION
            )
            return
        
        debug_config = ctx.config.get("ga4", {}).get("mp_debug", {})
        timeout = debug_config.get("timeout_ms", 5000) / 1000  # Convert to seconds
        
        validated_count = 0
        error_count = 0
        
        for request in ga4_requests:
            if not self._is_mp_request(request.url):
                continue  # Skip non-MP requests
            
            try:
                validation_result = await self._send_debug_request(request, timeout)
                self._process_debug_response(result, request, validation_result)
                validated_count += 1
                
            except Exception as e:
                error_count += 1
                result.add_warning_note(
                    f"MP debug validation failed for request: {str(e)}",
                    category=NoteCategory.VALIDATION,
                    request_url=request.url,
                    error=str(e)
                )
        
        # Add summary note
        if validated_count > 0:
            result.add_info_note(
                f"MP debug validation completed for {validated_count} requests",
                category=NoteCategory.VALIDATION,
                validated_requests=validated_count,
                failed_validations=error_count
            )
    
    def _is_mp_request(self, url: str) -> bool:
        """Check if request is a Measurement Protocol request."""
        return bool(match_url_pattern(url, "ga4_mp_collect"))
    
    async def _send_debug_request(self, request: RequestLog, timeout: float) -> Dict[str, Any]:
        """Send request to GA4 MP debug endpoint.
        
        Args:
            request: Original GA4 request to validate
            timeout: Request timeout in seconds
            
        Returns:
            Debug validation response
        """
        # Reconstruct debug URL
        debug_url = request.url.replace("/mp/collect", "/debug/mp/collect")
        
        # Prepare request data
        headers = {
            "Content-Type": request.request_headers.get("content-type", "application/json"),
            "User-Agent": "Tag-Sentinel-Debug-Validator/1.0.0"
        }
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            if request.method == "POST" and request.request_body:
                response = await client.post(
                    debug_url,
                    content=request.request_body,
                    headers=headers
                )
            else:
                response = await client.get(debug_url, headers=headers)
            
            response.raise_for_status()
            return response.json()
    
    def _process_debug_response(self, result: DetectResult, 
                               request: RequestLog, 
                               debug_response: Dict[str, Any]) -> None:
        """Process MP debug validation response.
        
        Args:
            result: Detection result to add notes to
            request: Original request
            debug_response: Debug endpoint response
        """
        validation_messages = debug_response.get("validationMessages", [])
        
        if not validation_messages:
            result.add_info_note(
                "GA4 MP request passed validation",
                category=NoteCategory.VALIDATION,
                request_url=request.url
            )
            return
        
        # Process validation errors
        error_messages = []
        warning_messages = []
        
        for message in validation_messages:
            message_type = message.get("validationType", "UNKNOWN")
            description = message.get("description", "Unknown validation error")
            field_path = message.get("fieldPath", "")
            
            formatted_message = f"{message_type}: {description}"
            if field_path:
                formatted_message += f" (field: {field_path})"
            
            if message_type in ["ERROR", "FATAL"]:
                error_messages.append(formatted_message)
            else:
                warning_messages.append(formatted_message)
        
        # Add error notes
        if error_messages:
            result.add_error_note(
                f"GA4 MP validation errors: {'; '.join(error_messages)}",
                category=NoteCategory.VALIDATION,
                request_url=request.url,
                validation_errors=error_messages
            )
        
        # Add warning notes
        if warning_messages:
            result.add_warning_note(
                f"GA4 MP validation warnings: {'; '.join(warning_messages)}",
                category=NoteCategory.VALIDATION,
                request_url=request.url,
                validation_warnings=warning_messages
            )