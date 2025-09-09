"""Utilities for analytics tag detection including regex patterns and parameter parsing."""

import json
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Pattern, Union
from urllib.parse import parse_qs, unquote_plus


class PatternLibrary:
    """Manages compiled regex patterns with caching and validation."""
    
    def __init__(self):
        self._patterns: Dict[str, Pattern[str]] = {}
        self._pattern_configs: Dict[str, Dict[str, Any]] = {}
    
    def add_pattern(self, name: str, pattern: str, flags: int = 0, 
                   description: str = "", category: str = "general") -> bool:
        """Add a compiled regex pattern to the library.
        
        Args:
            name: Unique name for the pattern
            pattern: Regex pattern string
            flags: Regex flags (re.IGNORECASE, etc.)
            description: Human-readable description
            category: Category for grouping patterns
            
        Returns:
            True if pattern was successfully compiled and added
        """
        try:
            compiled_pattern = re.compile(pattern, flags)
            self._patterns[name] = compiled_pattern
            self._pattern_configs[name] = {
                "pattern": pattern,
                "flags": flags,
                "description": description,
                "category": category
            }
            return True
        except re.error:
            return False
    
    def get_pattern(self, name: str) -> Optional[Pattern[str]]:
        """Get a compiled pattern by name."""
        return self._patterns.get(name)
    
    def match(self, pattern_name: str, text: str) -> Optional[re.Match[str]]:
        """Match text against a named pattern."""
        pattern = self.get_pattern(pattern_name)
        if pattern is None:
            return None
        return pattern.match(text)
    
    def search(self, pattern_name: str, text: str) -> Optional[re.Match[str]]:
        """Search text for a named pattern."""
        pattern = self.get_pattern(pattern_name)
        if pattern is None:
            return None
        return pattern.search(text)
    
    def findall(self, pattern_name: str, text: str) -> List[str]:
        """Find all matches of a named pattern in text."""
        pattern = self.get_pattern(pattern_name)
        if pattern is None:
            return []
        return pattern.findall(text)
    
    def test_pattern(self, name: str, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test a pattern against test cases.
        
        Args:
            name: Pattern name to test
            test_cases: List of dicts with 'input', 'should_match', and optional 'groups'
            
        Returns:
            Dict with test results
        """
        pattern = self.get_pattern(name)
        if pattern is None:
            return {"error": "Pattern not found", "passed": 0, "failed": 0}
        
        passed = 0
        failed = 0
        failures = []
        
        for i, case in enumerate(test_cases):
            input_text = case["input"]
            should_match = case["should_match"]
            expected_groups = case.get("groups", [])
            
            match = pattern.search(input_text)
            matched = match is not None
            
            if matched == should_match:
                if matched and expected_groups:
                    actual_groups = list(match.groups())
                    if actual_groups == expected_groups:
                        passed += 1
                    else:
                        failed += 1
                        failures.append({
                            "case": i,
                            "input": input_text,
                            "expected_groups": expected_groups,
                            "actual_groups": actual_groups
                        })
                else:
                    passed += 1
            else:
                failed += 1
                failures.append({
                    "case": i,
                    "input": input_text,
                    "expected_match": should_match,
                    "actual_match": matched
                })
        
        return {
            "passed": passed,
            "failed": failed,
            "total": len(test_cases),
            "failures": failures
        }
    
    def list_patterns(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all patterns with their metadata.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of pattern info dicts
        """
        patterns = []
        for name, config in self._pattern_configs.items():
            if category is None or config["category"] == category:
                patterns.append({
                    "name": name,
                    "pattern": config["pattern"],
                    "description": config["description"],
                    "category": config["category"]
                })
        return patterns
    
    def clear(self) -> None:
        """Clear all patterns."""
        self._patterns.clear()
        self._pattern_configs.clear()


# Global pattern library instance
patterns = PatternLibrary()


# Pre-compile common analytics patterns
def init_default_patterns():
    """Initialize default patterns for analytics detection."""
    
    # GA4 Patterns
    patterns.add_pattern(
        "ga4_mp_collect",
        r"https?://(?:www\.|region1\.)?google-analytics\.com/mp/collect",
        re.IGNORECASE,
        "GA4 Measurement Protocol collect endpoint",
        "ga4"
    )
    
    patterns.add_pattern(
        "ga4_g_collect", 
        r"https?://(?:www\.|region1\.)?google-analytics\.com/g/collect",
        re.IGNORECASE,
        "GA4 web collect endpoint (legacy)",
        "ga4"
    )
    
    patterns.add_pattern(
        "ga4_measurement_id",
        r"G-[A-Z0-9]{10}",
        0,
        "GA4 Measurement ID format",
        "ga4"
    )
    
    patterns.add_pattern(
        "ga4_regional_endpoint",
        r"https?://region\d+\.google-analytics\.com/mp/collect",
        re.IGNORECASE,
        "GA4 regional collect endpoints",
        "ga4"
    )
    
    # GTM Patterns
    patterns.add_pattern(
        "gtm_loader",
        r"https?://(?:www\.)?googletagmanager\.com/gtm\.js\?id=(GTM-[A-Z0-9]+)",
        re.IGNORECASE,
        "GTM container loader script",
        "gtm"
    )
    
    patterns.add_pattern(
        "gtm_container_id",
        r"GTM-[A-Z0-9]{7,}",
        0,
        "GTM Container ID format", 
        "gtm"
    )
    
    # General Analytics Patterns
    patterns.add_pattern(
        "google_analytics_domain",
        r"(?:www\.|region\d+\.)?google-analytics\.com",
        re.IGNORECASE,
        "Google Analytics domain patterns",
        "general"
    )
    
    patterns.add_pattern(
        "googletagmanager_domain",
        r"(?:www\.)?googletagmanager\.com",
        re.IGNORECASE,
        "Google Tag Manager domain",
        "general"
    )


@lru_cache(maxsize=128)
def compile_pattern(pattern: str, flags: int = 0) -> Optional[Pattern[str]]:
    """Compile and cache a regex pattern.
    
    Args:
        pattern: Regex pattern string
        flags: Regex compilation flags
        
    Returns:
        Compiled pattern or None if invalid
    """
    try:
        return re.compile(pattern, flags)
    except re.error:
        return None


def extract_url_components(url: str) -> Dict[str, str]:
    """Extract components from a URL for pattern matching.
    
    Args:
        url: URL to parse
        
    Returns:
        Dict with domain, path, query, fragment components
    """
    from urllib.parse import urlparse
    
    try:
        parsed = urlparse(url)
        return {
            "scheme": parsed.scheme,
            "domain": parsed.netloc.lower(),
            "path": parsed.path,
            "query": parsed.query,
            "fragment": parsed.fragment,
            "full_url": url
        }
    except Exception:
        return {
            "scheme": "",
            "domain": "",
            "path": "",
            "query": "",
            "fragment": "",
            "full_url": url
        }


def match_url_pattern(url: str, pattern_name: str) -> Optional[re.Match[str]]:
    """Match a URL against a named pattern.
    
    Args:
        url: URL to test
        pattern_name: Name of pattern in the pattern library
        
    Returns:
        Match object if pattern matches, None otherwise
    """
    return patterns.search(pattern_name, url)


def extract_measurement_id(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Extract GA4 measurement ID from URL or parameters.
    
    Args:
        url: Request URL
        params: Optional parsed parameters
        
    Returns:
        Measurement ID if found
    """
    # Try to find in URL path or query
    match = patterns.search("ga4_measurement_id", url)
    if match:
        return match.group(0)
    
    # Try to find in parameters if provided
    if params:
        # Check common parameter names
        for param_name in ["tid", "measurement_id", "id"]:
            if param_name in params:
                value = params[param_name]
                if isinstance(value, str):
                    match = patterns.search("ga4_measurement_id", value)
                    if match:
                        return match.group(0)
    
    return None


def extract_container_id(url: str) -> Optional[str]:
    """Extract GTM container ID from URL.
    
    Args:
        url: Request URL
        
    Returns:
        Container ID if found
    """
    match = patterns.search("gtm_container_id", url)
    return match.group(0) if match else None


def is_analytics_request(url: str) -> bool:
    """Check if URL appears to be an analytics request.
    
    Args:
        url: Request URL to check
        
    Returns:
        True if URL matches known analytics patterns
    """
    analytics_patterns = [
        "ga4_mp_collect",
        "ga4_g_collect", 
        "ga4_regional_endpoint",
        "gtm_loader"
    ]
    
    for pattern_name in analytics_patterns:
        if patterns.search(pattern_name, url):
            return True
    
    return False


def normalize_parameter_name(name: str) -> str:
    """Normalize parameter names for consistent analysis.
    
    Args:
        name: Parameter name to normalize
        
    Returns:
        Normalized parameter name
    """
    # Convert to lowercase and replace common variations
    normalized = name.lower()
    
    # Handle common GA4 parameter variations
    name_mappings = {
        "tid": "measurement_id",
        "tracking_id": "measurement_id",
        "cid": "client_id",
        "client-id": "client_id",
        "sid": "session_id",
        "session-id": "session_id",
        "dl": "page_location",
        "document-location": "page_location",
        "dt": "page_title",
        "document-title": "page_title"
    }
    
    return name_mappings.get(normalized, normalized)


def validate_measurement_id(measurement_id: str) -> bool:
    """Validate GA4 measurement ID format.
    
    Args:
        measurement_id: ID to validate
        
    Returns:
        True if valid format
    """
    if not measurement_id:
        return False
    
    match = patterns.match("ga4_measurement_id", measurement_id)
    return match is not None


def validate_container_id(container_id: str) -> bool:
    """Validate GTM container ID format.
    
    Args:
        container_id: ID to validate
        
    Returns:
        True if valid format  
    """
    if not container_id:
        return False
    
    match = patterns.match("gtm_container_id", container_id)
    return match is not None


class ParameterParser:
    """Utilities for parsing and normalizing request parameters from various formats."""
    
    @staticmethod
    def parse_json_payload(payload: str) -> Dict[str, Any]:
        """Parse JSON payload with error handling.
        
        Args:
            payload: JSON string to parse
            
        Returns:
            Parsed dict or empty dict if parsing fails
        """
        if not payload:
            return {}
        
        try:
            return json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return {}
    
    @staticmethod
    def parse_url_encoded(query_string: str) -> Dict[str, Any]:
        """Parse URL-encoded query string parameters.
        
        Args:
            query_string: Query string to parse (without leading ?)
            
        Returns:
            Dict of parameter name to value(s)
        """
        if not query_string:
            return {}
        
        try:
            # Remove leading ? if present
            if query_string.startswith('?'):
                query_string = query_string[1:]
            
            parsed = parse_qs(query_string, keep_blank_values=True)
            
            # Flatten single-item lists and decode values
            result = {}
            for key, values in parsed.items():
                decoded_key = unquote_plus(key)
                if len(values) == 1:
                    result[decoded_key] = unquote_plus(values[0])
                else:
                    result[decoded_key] = [unquote_plus(v) for v in values]
            
            return result
        except Exception:
            return {}
    
    @staticmethod
    def parse_form_data(body: str, content_type: str = "") -> Dict[str, Any]:
        """Parse form data from request body.
        
        Args:
            body: Request body string
            content_type: Content-Type header value
            
        Returns:
            Parsed parameters dict
        """
        if not body:
            return {}
        
        # Handle application/x-www-form-urlencoded
        if "application/x-www-form-urlencoded" in content_type.lower():
            return ParameterParser.parse_url_encoded(body)
        
        # Handle application/json
        if "application/json" in content_type.lower():
            return ParameterParser.parse_json_payload(body)
        
        # Try to parse as URL-encoded by default
        return ParameterParser.parse_url_encoded(body)
    
    @staticmethod
    def extract_ga4_events(params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract GA4 events from parsed parameters.
        
        Args:
            params: Parsed parameters dict
            
        Returns:
            List of event dicts with names and parameters
        """
        events = []
        
        # Handle Measurement Protocol format (array of events)
        if "events" in params and isinstance(params["events"], list):
            for event in params["events"]:
                if isinstance(event, dict) and "name" in event:
                    events.append({
                        "name": event["name"],
                        "params": event.get("params", {}),
                        "custom_params": event.get("custom_parameters", {})
                    })
        
        # Handle single event format
        elif "en" in params:  # Event name parameter
            events.append({
                "name": params["en"],
                "params": {k: v for k, v in params.items() if k != "en"},
                "custom_params": {}
            })
        
        # Handle legacy gtag format
        elif "t" in params and params["t"] == "event" and "ea" in params:
            events.append({
                "name": params.get("ea", "unknown"),
                "params": params,
                "custom_params": {}
            })
        
        return events
    
    @staticmethod
    def normalize_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameter names and values for consistent analysis.
        
        Args:
            params: Raw parameters dict
            
        Returns:
            Dict with normalized parameter names and values
        """
        normalized = {}
        
        for key, value in params.items():
            # Normalize the key
            normalized_key = normalize_parameter_name(key)
            
            # Normalize the value
            if isinstance(value, str):
                # Decode common URL encoding
                try:
                    normalized_value = unquote_plus(value)
                except Exception:
                    normalized_value = value
            else:
                normalized_value = value
            
            normalized[normalized_key] = normalized_value
        
        return normalized
    
    @staticmethod
    def extract_client_info(params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract client identification information from parameters.
        
        Args:
            params: Parsed parameters dict
            
        Returns:
            Dict with client_id, session_id, user_id if found
        """
        client_info = {}
        
        # Common client ID parameter names
        client_id_params = ["cid", "client_id", "_cid", "clientId"]
        for param in client_id_params:
            if param in params:
                client_info["client_id"] = params[param]
                break
        
        # Common session ID parameter names  
        session_id_params = ["sid", "session_id", "_sid", "sessionId"]
        for param in session_id_params:
            if param in params:
                client_info["session_id"] = params[param]
                break
        
        # User ID parameters
        user_id_params = ["uid", "user_id", "_uid", "userId"]
        for param in user_id_params:
            if param in params:
                client_info["user_id"] = params[param]
                break
        
        return client_info
    
    @staticmethod
    def extract_page_info(params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract page information from parameters.
        
        Args:
            params: Parsed parameters dict
            
        Returns:
            Dict with page_location, page_title, page_referrer if found
        """
        page_info = {}
        
        # Page location/URL
        location_params = ["dl", "page_location", "location", "url"]
        for param in location_params:
            if param in params:
                page_info["page_location"] = params[param]
                break
        
        # Page title
        title_params = ["dt", "page_title", "title"]
        for param in title_params:
            if param in params:
                page_info["page_title"] = params[param]
                break
        
        # Page referrer
        referrer_params = ["dr", "page_referrer", "referrer", "ref"]
        for param in referrer_params:
            if param in params:
                page_info["page_referrer"] = params[param]
                break
        
        return page_info
    
    @staticmethod
    def extract_custom_dimensions(params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract custom dimensions and metrics from parameters.
        
        Args:
            params: Parsed parameters dict
            
        Returns:
            Dict with custom_dimensions and custom_metrics
        """
        custom_data = {
            "custom_dimensions": {},
            "custom_metrics": {}
        }
        
        for key, value in params.items():
            # GA4 custom parameters (often prefixed with custom_)
            if key.startswith("custom_"):
                custom_data["custom_dimensions"][key] = value
            
            # Universal Analytics custom dimensions (cd1, cd2, etc.)
            elif key.startswith("cd") and key[2:].isdigit():
                dimension_index = key[2:]
                custom_data["custom_dimensions"][f"custom_dimension_{dimension_index}"] = value
            
            # Universal Analytics custom metrics (cm1, cm2, etc.)
            elif key.startswith("cm") and key[2:].isdigit():
                metric_index = key[2:]
                custom_data["custom_metrics"][f"custom_metric_{metric_index}"] = value
        
        return custom_data
    
    @staticmethod
    def validate_required_params(params: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
        """Validate that required parameters are present.
        
        Args:
            params: Parameters dict to validate
            required: List of required parameter names
            
        Returns:
            Dict with validation results
        """
        result = {
            "valid": True,
            "missing_params": [],
            "present_params": []
        }
        
        for param in required:
            if param in params and params[param] is not None:
                result["present_params"].append(param)
            else:
                result["missing_params"].append(param)
                result["valid"] = False
        
        return result
    
    @staticmethod
    def sanitize_sensitive_params(params: Dict[str, Any], 
                                sensitive_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Remove or redact sensitive parameters.
        
        Args:
            params: Parameters dict to sanitize
            sensitive_patterns: List of regex patterns for sensitive parameter names
            
        Returns:
            Sanitized parameters dict
        """
        if sensitive_patterns is None:
            # Default sensitive patterns
            sensitive_patterns = [
                r".*password.*",
                r".*token.*", 
                r".*key.*",
                r".*secret.*",
                r".*auth.*",
                r".*credential.*"
            ]
        
        sanitized = {}
        
        for key, value in params.items():
            is_sensitive = False
            
            for pattern in sensitive_patterns:
                if re.match(pattern, key.lower()):
                    is_sensitive = True
                    break
            
            if is_sensitive:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        
        return sanitized


# DataLayer validation utilities
def validate_datalayer_structure(datalayer_data: Any) -> Dict[str, Any]:
    """Validate dataLayer object structure and content.
    
    Args:
        datalayer_data: Raw dataLayer data (usually a list)
        
    Returns:
        Validation result with structure analysis
    """
    result = {
        "is_valid": False,
        "is_array": False,
        "total_events": 0,
        "event_types": set(),
        "has_gtm_events": False,
        "has_ecommerce": False,
        "structure_issues": [],
        "common_variables": set(),
        "first_event_timestamp": None,
        "last_event_timestamp": None
    }
    
    try:
        # Check if dataLayer exists and is an array
        if not isinstance(datalayer_data, list):
            result["structure_issues"].append("dataLayer is not an array")
            return result
        
        result["is_array"] = True
        result["total_events"] = len(datalayer_data)
        
        if len(datalayer_data) == 0:
            result["structure_issues"].append("dataLayer is empty")
            return result
        
        # Analyze each event in the dataLayer
        for i, event in enumerate(datalayer_data):
            if not isinstance(event, dict):
                result["structure_issues"].append(f"Event at index {i} is not an object")
                continue
            
            # Track event types
            if "event" in event:
                event_type = event["event"]
                result["event_types"].add(event_type)
                
                # Check for GTM-specific events
                gtm_events = ["gtm.js", "gtm.dom", "gtm.load", "gtm.click", "gtm.scroll"]
                if event_type in gtm_events or event_type.startswith("gtm."):
                    result["has_gtm_events"] = True
            
            # Check for ecommerce data
            if "ecommerce" in event and event["ecommerce"]:
                result["has_ecommerce"] = True
            
            # Track common variables
            for key in event.keys():
                if key not in ["event", "gtm.uniqueEventId", "gtm.start"]:
                    result["common_variables"].add(key)
            
            # Track timestamps if available
            if "gtm.start" in event:
                timestamp = event["gtm.start"]
                if result["first_event_timestamp"] is None:
                    result["first_event_timestamp"] = timestamp
                result["last_event_timestamp"] = timestamp
        
        # If we made it here without major issues, it's valid
        if not result["structure_issues"]:
            result["is_valid"] = True
            
    except Exception as e:
        result["structure_issues"].append(f"Error parsing dataLayer: {str(e)}")
    
    # Convert sets to lists for JSON serialization
    result["event_types"] = list(result["event_types"])
    result["common_variables"] = list(result["common_variables"])
    
    return result


def extract_datalayer_events(datalayer_data: List[Dict[str, Any]], 
                           event_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Extract specific events from dataLayer.
    
    Args:
        datalayer_data: dataLayer array
        event_filter: Optional event type filter (e.g., "purchase", "gtm.")
        
    Returns:
        List of matching events
    """
    if not isinstance(datalayer_data, list):
        return []
    
    events = []
    for event in datalayer_data:
        if not isinstance(event, dict):
            continue
        
        if "event" not in event:
            continue
        
        event_type = event["event"]
        
        if event_filter is None:
            events.append(event)
        elif event_filter.endswith("."):
            # Prefix match (e.g., "gtm.")
            if event_type.startswith(event_filter):
                events.append(event)
        else:
            # Exact match
            if event_type == event_filter:
                events.append(event)
    
    return events


def check_datalayer_best_practices(validation_result: Dict[str, Any]) -> List[Dict[str, str]]:
    """Check dataLayer against best practices.
    
    Args:
        validation_result: Result from validate_datalayer_structure
        
    Returns:
        List of best practice recommendations
    """
    recommendations = []
    
    if not validation_result["is_valid"]:
        recommendations.append({
            "type": "error",
            "message": "dataLayer has structural issues that prevent analysis",
            "category": "structure"
        })
        return recommendations
    
    # Check for basic GTM initialization
    if not validation_result["has_gtm_events"]:
        recommendations.append({
            "type": "warning", 
            "message": "No GTM initialization events found (gtm.js, gtm.dom, gtm.load)",
            "category": "initialization"
        })
    
    # Check for ecommerce tracking
    if validation_result["total_events"] > 5 and not validation_result["has_ecommerce"]:
        recommendations.append({
            "type": "info",
            "message": "No ecommerce events detected - consider implementing enhanced ecommerce",
            "category": "ecommerce"
        })
    
    # Check event volume
    if validation_result["total_events"] > 100:
        recommendations.append({
            "type": "warning",
            "message": f"High number of dataLayer events ({validation_result['total_events']}) may impact performance",
            "category": "performance"
        })
    elif validation_result["total_events"] < 3:
        recommendations.append({
            "type": "info", 
            "message": "Very few dataLayer events - ensure tracking is properly implemented",
            "category": "completeness"
        })
    
    # Check for custom event diversity
    custom_events = [et for et in validation_result["event_types"] 
                    if not et.startswith("gtm.")]
    
    if len(custom_events) == 0:
        recommendations.append({
            "type": "info",
            "message": "No custom events found - consider tracking user interactions",
            "category": "events"
        })
    
    return recommendations


# Initialize default patterns on module import
init_default_patterns()