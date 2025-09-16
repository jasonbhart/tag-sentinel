"""Google Tag Manager (GTM) detector for container loading and dataLayer validation."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from urllib.parse import parse_qs, urlparse

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
    resilient_operation
)
from .utils import (
    patterns,
    extract_container_id,
    extract_url_components,
    match_url_pattern,
    validate_container_id,
    validate_datalayer_structure,
    extract_datalayer_events,
    check_datalayer_best_practices
)
from ..models.capture import PageResult, RequestLog


class GTMDetector(BaseDetector, ResilientDetector):
    """Detector for Google Tag Manager container loading and configuration."""
    
    def __init__(self, name: str = "GTMDetector"):
        super().__init__(name, "1.0.0")
    
    @property
    def supported_vendors(self) -> Set[Vendor]:
        """GTM detector supports GTM vendor.""" 
        return {Vendor.GTM}
    
    def detect(self, page: PageResult, ctx: DetectContext) -> DetectResult:
        """Analyze page for GTM container activity.
        
        Args:
            page: Page capture result with network requests
            ctx: Detection context with configuration
            
        Returns:
            Detection results with GTM events and notes
        """
        result = self._create_result()
        start_time = datetime.utcnow()
        
        try:
            # Find GTM container loading requests
            gtm_requests = self._find_gtm_requests(page.network_requests)
            result.processed_requests = len(gtm_requests)
            
            # Extract container information and create events
            containers = self._extract_container_info(gtm_requests, page.url)
            
            for container in containers:
                event = self._create_container_event(container, page.url, ctx)
                result.add_event(event)
            
            # Validate dataLayer if enabled in context
            if ctx.config.get("gtm", {}).get("validate_datalayer", True):
                self._validate_datalayer(result, page, ctx)
            
            # Add analysis notes
            self._add_analysis_notes(result, page.url, containers, ctx)
            
            # Set page URL for all notes
            self._set_note_page_urls(result, page.url)
            
        except Exception as e:
            result.success = False
            result.error_message = f"GTM detection failed: {str(e)}"
            result.add_error_note(
                f"GTM detector encountered an error: {str(e)}",
                category=NoteCategory.VALIDATION
            )
        
        # Calculate processing time
        end_time = datetime.utcnow()
        result.processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        return result
    
    def _find_gtm_requests(self, requests: List[RequestLog]) -> List[RequestLog]:
        """Find all requests that match GTM patterns.
        
        Args:
            requests: List of network requests to analyze
            
        Returns:
            List of requests that match GTM loader patterns
        """
        gtm_requests = []
        
        for request in requests:
            if self._is_gtm_request(request.url):
                gtm_requests.append(request)
        
        return gtm_requests
    
    def _is_gtm_request(self, url: str) -> bool:
        """Check if URL matches GTM loader patterns.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL matches GTM patterns
        """
        return bool(match_url_pattern(url, "gtm_loader"))
    
    def _extract_container_info(self, gtm_requests: List[RequestLog], 
                               page_url: str) -> List[Dict[str, Any]]:
        """Extract container information from GTM requests.
        
        Args:
            gtm_requests: List of GTM loader requests
            page_url: Page URL for context
            
        Returns:
            List of container info dictionaries
        """
        containers = []
        seen_containers = set()  # Track unique containers
        
        for request in gtm_requests:
            container_info = self._analyze_container_request(request, page_url)
            
            # Avoid duplicates based on container ID
            container_id = container_info.get("container_id")
            if container_id and container_id not in seen_containers:
                containers.append(container_info)
                seen_containers.add(container_id)
            elif not container_id:
                # Still include requests without container ID for error reporting
                containers.append(container_info)
        
        return containers
    
    @resilient_operation("analyze_container", ErrorSeverity.MEDIUM, ErrorCategory.PARSING, {})
    def _analyze_container_request(self, request: RequestLog, 
                                  page_url: str) -> Dict[str, Any]:
        """Analyze a single GTM container request.
        
        Args:
            request: GTM loader request
            page_url: Page URL for context
            
        Returns:
            Container information dictionary
        """
        container_info = {
            "request_url": request.url,
            "request_method": request.method,
            "status_code": request.status_code,
            "timing_ms": self._calculate_timing(request),
            "load_successful": request.is_successful,
            "request_headers": dict(request.request_headers),
            "response_headers": dict(request.response_headers)
        }
        
        # Extract container ID from URL
        container_id = extract_container_id(request.url)
        container_info["container_id"] = container_id
        
        # Validate container ID format
        if container_id:
            container_info["container_id_valid"] = validate_container_id(container_id)
        else:
            container_info["container_id_valid"] = False
        
        # Parse additional URL parameters
        url_params = self._parse_gtm_url_parameters(request.url)
        container_info.update(url_params)
        
        # Analyze response if available
        if request.response_body:
            response_analysis = self._analyze_gtm_response(request.response_body)
            container_info["response_analysis"] = response_analysis
        
        # Determine container configuration from URL
        config_info = self._extract_container_config(request.url)
        container_info.update(config_info)
        
        return container_info
    
    def _parse_gtm_url_parameters(self, url: str) -> Dict[str, Any]:
        """Parse parameters from GTM loader URL.
        
        Args:
            url: GTM loader URL
            
        Returns:
            Dict of parsed parameters
        """
        parsed_url = urlparse(url)
        params = {}
        
        if parsed_url.query:
            query_params = parse_qs(parsed_url.query, keep_blank_values=True)
            
            # Flatten single-value lists
            for key, values in query_params.items():
                if len(values) == 1:
                    params[key] = values[0]
                else:
                    params[key] = values
        
        return {
            "url_parameters": params,
            "gtm_auth": params.get("gtm_auth"),
            "gtm_preview": params.get("gtm_preview"), 
            "gtm_cookies_win": params.get("gtm_cookies_win"),
            "cb": params.get("cb"),  # Cache buster
        }
    
    def _analyze_gtm_response(self, response_body: str) -> Dict[str, Any]:
        """Analyze GTM container response content.
        
        Args:
            response_body: Response body content
            
        Returns:
            Analysis results
        """
        analysis = {
            "content_length": len(response_body),
            "contains_gtag": "gtag(" in response_body,
            "contains_dataLayer": "dataLayer" in response_body,
            "contains_google_analytics": "google-analytics.com" in response_body,
            "contains_gtm_js": "googletagmanager.com" in response_body,
            "minified": not ("\n" in response_body and "  " in response_body),
        }
        
        # Look for common GTM functions and variables
        gtm_indicators = [
            "window.google_tag_manager",
            "gtm.js",
            "gtm.load", 
            "gtag.config",
            "ga.create",
            "fbq("
        ]
        
        found_indicators = []
        for indicator in gtm_indicators:
            if indicator in response_body:
                found_indicators.append(indicator)
        
        analysis["gtm_indicators"] = found_indicators
        analysis["gtm_indicators_count"] = len(found_indicators)
        
        return analysis
    
    def _extract_container_config(self, url: str) -> Dict[str, Any]:
        """Extract container configuration from URL using parsed parameters.

        Args:
            url: GTM loader URL

        Returns:
            Configuration information
        """
        config = {}

        # Parse URL parameters to avoid false positives from substring checks
        parsed_params = self._parse_gtm_url_parameters(url)

        # Check for environment/workspace parameters using parsed data
        if parsed_params.get("gtm_auth"):
            config["has_workspace_auth"] = True
            config["environment_type"] = "workspace"
        else:
            config["has_workspace_auth"] = False
            config["environment_type"] = "live"

        # Check for preview mode using parsed data
        if parsed_params.get("gtm_preview"):
            config["preview_mode"] = True
        else:
            config["preview_mode"] = False

        # Check for custom domain
        parsed_url = urlparse(url)
        if parsed_url.netloc != "www.googletagmanager.com":
            config["custom_domain"] = parsed_url.netloc
        else:
            config["custom_domain"] = None

        return config
    
    def _create_container_event(self, container_info: Dict[str, Any], 
                              page_url: str, ctx: DetectContext) -> TagEvent:
        """Create a TagEvent from container information.
        
        Args:
            container_info: Container analysis results
            page_url: Page URL
            ctx: Detection context
            
        Returns:
            TagEvent for the container
        """
        container_id = container_info.get("container_id")
        load_successful = container_info.get("load_successful", False)
        
        # Determine event status
        if load_successful and container_id and container_info.get("container_id_valid"):
            status = TagStatus.OK
            confidence = Confidence.HIGH
        elif load_successful and container_id:
            status = TagStatus.OK  
            confidence = Confidence.MEDIUM
        elif load_successful:
            status = TagStatus.OK
            confidence = Confidence.LOW
        else:
            status = TagStatus.ERROR
            confidence = Confidence.LOW
        
        # Build event parameters
        event_params = {
            "container_id": container_id,
            "container_id_valid": container_info.get("container_id_valid", False),
            "environment_type": container_info.get("environment_type", "unknown"),
            "preview_mode": container_info.get("preview_mode", False),
            "custom_domain": container_info.get("custom_domain"),
            "request_url": container_info["request_url"],
            "status_code": container_info.get("status_code"),
            "timing_ms": container_info.get("timing_ms"),
            "has_workspace_auth": container_info.get("has_workspace_auth", False)
        }
        
        # Include response analysis if available
        if "response_analysis" in container_info:
            response_analysis = container_info["response_analysis"]
            event_params.update({
                "content_length": response_analysis.get("content_length"),
                "contains_dataLayer": response_analysis.get("contains_dataLayer"),
                "contains_gtag": response_analysis.get("contains_gtag"),
                "gtm_indicators_count": response_analysis.get("gtm_indicators_count", 0),
                "minified": response_analysis.get("minified")
            })
        
        # Remove None values
        event_params = {k: v for k, v in event_params.items() if v is not None}
        
        return TagEvent(
            vendor=Vendor.GTM,
            name="container_load",
            category="container",
            id=container_id,
            page_url=page_url,
            request_url=container_info["request_url"],
            timing_ms=container_info.get("timing_ms"),
            status=status,
            confidence=confidence,
            params=event_params,
            detection_method="network_request",
            detector_version=self.version
        )
    
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
                          containers: List[Dict[str, Any]], ctx: DetectContext) -> None:
        """Add analysis notes based on GTM container detection.
        
        Args:
            result: Detection result to add notes to
            page_url: Page URL
            containers: List of detected containers
            ctx: Detection context
        """
        if not containers:
            result.add_info_note(
                "No GTM containers detected on this page",
                category=NoteCategory.DATA_QUALITY
            )
            return
        
        # Note about multiple containers
        valid_containers = [c for c in containers if c.get("container_id")]
        if len(valid_containers) > 1:
            container_ids = [c["container_id"] for c in valid_containers]
            result.add_warning_note(
                f"Multiple GTM containers detected: {container_ids}",
                category=NoteCategory.CONFIGURATION,
                container_ids=container_ids
            )
        
        # Check for invalid container IDs
        invalid_containers = [c for c in containers
                            if c.get("container_id") and not c.get("container_id_valid")]
        if invalid_containers:
            invalid_ids = [c["container_id"] for c in invalid_containers]
            result.add_warning_note(
                f"Invalid GTM container ID format detected: {invalid_ids}",
                category=NoteCategory.VALIDATION,
                invalid_container_ids=invalid_ids
            )

        # Check for unexpected container IDs if expected IDs are configured
        gtm_config = ctx.config.get("gtm", {})
        expected_ids = gtm_config.get("expected_container_ids", [])
        if expected_ids:
            actual_ids = [c["container_id"] for c in valid_containers]
            unexpected_ids = [id for id in actual_ids if id not in expected_ids]
            missing_ids = [id for id in expected_ids if id not in actual_ids]

            if unexpected_ids:
                result.add_warning_note(
                    f"Unexpected GTM container IDs detected: {unexpected_ids}",
                    category=NoteCategory.CONFIGURATION,
                    unexpected_container_ids=unexpected_ids,
                    expected_container_ids=expected_ids
                )

            if missing_ids:
                result.add_warning_note(
                    f"Expected GTM container IDs not found: {missing_ids}",
                    category=NoteCategory.CONFIGURATION,
                    missing_container_ids=missing_ids,
                    expected_container_ids=expected_ids
                )
        
        # Check for failed container loads
        failed_containers = [c for c in containers if not c.get("load_successful")]
        if failed_containers:
            failed_urls = [c["request_url"] for c in failed_containers]
            result.add_error_note(
                f"{len(failed_containers)} GTM container requests failed to load",
                category=NoteCategory.VALIDATION,
                failed_count=len(failed_containers),
                failed_urls=failed_urls
            )
        
        # Check for preview mode
        preview_containers = [c for c in containers if c.get("preview_mode")]
        if preview_containers:
            result.add_info_note(
                "GTM preview mode detected - this is typically used for testing",
                category=NoteCategory.CONFIGURATION,
                preview_containers=len(preview_containers)
            )
        
        # Check for workspace authentication
        workspace_containers = [c for c in containers if c.get("has_workspace_auth")]
        if workspace_containers:
            result.add_info_note(
                "GTM workspace authentication detected - this indicates a non-live environment",
                category=NoteCategory.CONFIGURATION,
                workspace_containers=len(workspace_containers)
            )
        
        # Performance warnings for slow loads
        slow_containers = [c for c in containers 
                          if c.get("timing_ms") and c["timing_ms"] > 3000]
        if slow_containers:
            result.add_warning_note(
                f"{len(slow_containers)} GTM containers took longer than 3 seconds to load",
                category=NoteCategory.PERFORMANCE,
                slow_containers=[c["container_id"] for c in slow_containers if c.get("container_id")]
            )
        
        # Check for dataLayer presence in responses
        containers_with_datalayer = [c for c in containers 
                                   if c.get("response_analysis", {}).get("contains_dataLayer")]
        containers_without_datalayer = [c for c in containers 
                                      if c.get("response_analysis", {}).get("contains_dataLayer") is False]
        
        if containers_without_datalayer:
            result.add_warning_note(
                "GTM container loaded but no dataLayer reference found in response",
                category=NoteCategory.DATA_QUALITY,
                containers_without_datalayer=len(containers_without_datalayer)
            )
        
        # Summary note
        total_containers = len(valid_containers)
        successful_containers = len([c for c in valid_containers if c.get("load_successful")])
        
        result.add_info_note(
            f"GTM detection summary: {successful_containers}/{total_containers} containers loaded successfully",
            category=NoteCategory.DATA_QUALITY,
            total_containers=total_containers,
            successful_containers=successful_containers
        )
    
    @resilient_operation("validate_datalayer", ErrorSeverity.MEDIUM, ErrorCategory.VALIDATION, None)
    def _validate_datalayer(self, result: DetectResult, page: PageResult, ctx: DetectContext) -> None:
        """Validate dataLayer presence and structure.
        
        Args:
            result: Detection result to add findings to
            page: Page capture result
            ctx: Detection context
        """
        try:
            # Look for dataLayer in network requests (JavaScript responses)
            datalayer_data = self._extract_datalayer_from_responses(page.network_requests)
            
            if not datalayer_data:
                # No dataLayer found in network responses
                result.add_warning_note(
                    "No dataLayer object found in page responses",
                    category=NoteCategory.DATA_QUALITY,
                    recommendation="Implement dataLayer for GTM event tracking"
                )
                return
            
            # Validate dataLayer structure
            validation_result = validate_datalayer_structure(datalayer_data)
            
            if not validation_result["is_valid"]:
                # Report structural issues
                issues = validation_result.get("structure_issues", [])
                result.add_error_note(
                    f"dataLayer validation failed: {'; '.join(issues)}",
                    category=NoteCategory.VALIDATION,
                    structure_issues=issues
                )
                return
            
            # Create dataLayer validation event
            datalayer_event = self._create_datalayer_event(validation_result, page.url, ctx)
            result.add_event(datalayer_event)
            
            # Check best practices
            recommendations = check_datalayer_best_practices(validation_result)
            
            for rec in recommendations:
                if rec["type"] == "error":
                    result.add_error_note(
                        rec["message"],
                        category=NoteCategory.VALIDATION,
                        recommendation_category=rec["category"]
                    )
                elif rec["type"] == "warning":
                    result.add_warning_note(
                        rec["message"], 
                        category=NoteCategory.CONFIGURATION,
                        recommendation_category=rec["category"]
                    )
                else:  # info
                    result.add_info_note(
                        rec["message"],
                        category=NoteCategory.DATA_QUALITY,
                        recommendation_category=rec["category"]
                    )
            
            # Add summary note
            result.add_info_note(
                f"dataLayer validation completed: {validation_result['total_events']} events, "
                f"{len(validation_result['event_types'])} event types",
                category=NoteCategory.DATA_QUALITY,
                total_events=validation_result["total_events"],
                event_types=len(validation_result["event_types"]),
                has_gtm_events=validation_result["has_gtm_events"],
                has_ecommerce=validation_result["has_ecommerce"]
            )
            
        except Exception as e:
            result.add_error_note(
                f"dataLayer validation failed with error: {str(e)}",
                category=NoteCategory.VALIDATION,
                error=str(e)
            )
    
    def _extract_datalayer_from_responses(self, requests: List[RequestLog]) -> Optional[List[Dict[str, Any]]]:
        """Extract dataLayer data from network request responses.
        
        Args:
            requests: List of network requests
            
        Returns:
            dataLayer data if found, None otherwise
        """
        # Look for JavaScript responses that might contain dataLayer initialization
        for request in requests:
            if not request.response_body:
                continue
            
            # Check HTML responses for inline dataLayer
            if "text/html" in request.response_headers.get("content-type", ""):
                datalayer = self._extract_datalayer_from_html(request.response_body)
                if datalayer is not None:
                    return datalayer
            
            # Check JavaScript responses for dataLayer assignments
            elif ("javascript" in request.response_headers.get("content-type", "") or 
                  request.url.endswith(".js")):
                datalayer = self._extract_datalayer_from_js(request.response_body)
                if datalayer is not None:
                    return datalayer
        
        return None
    
    def _extract_datalayer_from_html(self, html_content: str) -> Optional[List[Dict[str, Any]]]:
        """Extract dataLayer from HTML content.
        
        Args:
            html_content: HTML response content
            
        Returns:
            dataLayer data if found
        """
        import re
        import json
        
        # Look for dataLayer initialization patterns
        patterns = [
            r'window\.dataLayer\s*=\s*(\[[^;]*\]);',
            r'dataLayer\s*=\s*(\[[^;]*\]);',
            r'var\s+dataLayer\s*=\s*(\[[^;]*\]);',
            r'let\s+dataLayer\s*=\s*(\[[^;]*\]);',
            r'const\s+dataLayer\s*=\s*(\[[^;]*\]);'
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, html_content, re.DOTALL)
            if matches:
                try:
                    datalayer_str = matches.group(1)
                    # Simple JSON parse - may fail for complex JS expressions
                    datalayer_data = json.loads(datalayer_str)
                    if isinstance(datalayer_data, list):
                        return datalayer_data
                except (json.JSONDecodeError, ValueError):
                    continue
        
        return None
    
    def _extract_datalayer_from_js(self, js_content: str) -> Optional[List[Dict[str, Any]]]:
        """Extract dataLayer from JavaScript content.
        
        Args:
            js_content: JavaScript response content
            
        Returns:
            dataLayer data if found
        """
        # For JavaScript files, look for dataLayer.push() calls or initializations
        # This is a simplified extraction - a full implementation would need 
        # a JavaScript parser for complex cases
        
        if "dataLayer" not in js_content:
            return None
        
        # Look for simple dataLayer initialization
        import re
        import json
        
        init_pattern = r'dataLayer\s*=\s*(\[[^;]*\]);'
        match = re.search(init_pattern, js_content)
        if match:
            try:
                datalayer_str = match.group(1)
                datalayer_data = json.loads(datalayer_str)
                if isinstance(datalayer_data, list):
                    return datalayer_data
            except (json.JSONDecodeError, ValueError):
                pass
        
        # If no initialization found but dataLayer is referenced,
        # return empty array to indicate presence
        return []
    
    def _create_datalayer_event(self, validation_result: Dict[str, Any], 
                              page_url: str, ctx: DetectContext) -> TagEvent:
        """Create a TagEvent for dataLayer validation results.
        
        Args:
            validation_result: Validation results from validate_datalayer_structure
            page_url: Page URL
            ctx: Detection context
            
        Returns:
            TagEvent for dataLayer validation
        """
        # Determine status and confidence based on validation
        if validation_result["is_valid"]:
            if validation_result["has_gtm_events"] and validation_result["total_events"] >= 3:
                status = TagStatus.OK
                confidence = Confidence.HIGH
            elif validation_result["total_events"] > 0:
                status = TagStatus.OK
                confidence = Confidence.MEDIUM
            else:
                status = TagStatus.OK
                confidence = Confidence.LOW
        else:
            status = TagStatus.ERROR
            confidence = Confidence.LOW
        
        # Build event parameters
        event_params = {
            "is_valid": validation_result["is_valid"],
            "is_array": validation_result["is_array"],
            "total_events": validation_result["total_events"],
            "event_types_count": len(validation_result["event_types"]),
            "event_types": validation_result["event_types"][:10],  # Limit for size
            "has_gtm_events": validation_result["has_gtm_events"],
            "has_ecommerce": validation_result["has_ecommerce"],
            "common_variables_count": len(validation_result["common_variables"]),
            "structure_issues": validation_result.get("structure_issues", [])
        }
        
        # Add timing information if available
        if validation_result.get("first_event_timestamp"):
            event_params["first_event_timestamp"] = validation_result["first_event_timestamp"]
        if validation_result.get("last_event_timestamp"):
            event_params["last_event_timestamp"] = validation_result["last_event_timestamp"]
        
        return TagEvent(
            vendor=Vendor.GTM,
            name="dataLayer_validation",
            category="datalayer",
            page_url=page_url,
            status=status,
            confidence=confidence,
            params=event_params,
            detection_method="response_analysis",
            detector_version=self.version
        )