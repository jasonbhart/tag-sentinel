"""Unit tests for detector utilities."""

import pytest
import re

from app.audit.detectors.utils import (
    PatternLibrary,
    ParameterParser,
    patterns,
    extract_measurement_id,
    extract_container_id,
    is_analytics_request,
    normalize_parameter_name,
    validate_measurement_id,
    validate_container_id
)


class TestPatternLibrary:
    """Test PatternLibrary functionality."""
    
    def setup_method(self):
        """Set up test with fresh pattern library."""
        self.lib = PatternLibrary()
    
    def test_add_pattern_valid(self):
        """Test adding valid regex pattern."""
        success = self.lib.add_pattern(
            "test_pattern", 
            r"\d+", 
            0, 
            "Match digits", 
            "test"
        )
        assert success is True
        
        pattern = self.lib.get_pattern("test_pattern")
        assert pattern is not None
        assert pattern.match("123") is not None
        assert pattern.match("abc") is None
    
    def test_add_pattern_invalid(self):
        """Test adding invalid regex pattern."""
        success = self.lib.add_pattern(
            "invalid_pattern",
            r"[invalid regex(",  # Invalid regex
            0,
            "Invalid pattern",
            "test"
        )
        assert success is False
        
        pattern = self.lib.get_pattern("invalid_pattern")
        assert pattern is None
    
    def test_match_search_findall(self):
        """Test pattern matching methods."""
        self.lib.add_pattern("email", r"([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})")
        
        # Test match (from beginning)
        match = self.lib.match("email", "user@example.com")
        assert match is not None
        
        # Test search (anywhere in string)  
        search = self.lib.search("email", "Contact us at user@example.com for help")
        assert search is not None
        
        # Test findall
        results = self.lib.findall("email", "user1@example.com and user2@test.org")
        assert len(results) == 2
    
    def test_pattern_testing(self):
        """Test pattern validation with test cases."""
        self.lib.add_pattern("number", r"\d+")
        
        test_cases = [
            {"input": "123", "should_match": True},
            {"input": "abc", "should_match": False},
            {"input": "12a", "should_match": True},  # Partial match
            {"input": "", "should_match": False}
        ]
        
        results = self.lib.test_pattern("number", test_cases)
        
        assert results["total"] == 4
        assert results["passed"] == 4
        assert results["failed"] == 0
    
    def test_list_patterns(self):
        """Test listing patterns with category filter."""
        self.lib.add_pattern("ga4_pattern", r"G-\w+", description="GA4 ID", category="ga4")
        self.lib.add_pattern("gtm_pattern", r"GTM-\w+", description="GTM ID", category="gtm")
        
        # List all patterns
        all_patterns = self.lib.list_patterns()
        assert len(all_patterns) == 2
        
        # Filter by category
        ga4_patterns = self.lib.list_patterns("ga4")
        assert len(ga4_patterns) == 1
        assert ga4_patterns[0]["name"] == "ga4_pattern"
    
    def test_clear(self):
        """Test clearing all patterns."""
        self.lib.add_pattern("test", r"\d+")
        assert len(self.lib.list_patterns()) == 1
        
        self.lib.clear()
        assert len(self.lib.list_patterns()) == 0


class TestParameterParser:
    """Test ParameterParser functionality."""
    
    def setup_method(self):
        """Set up test with ParameterParser."""
        self.parser = ParameterParser()
    
    def test_parse_json_payload_valid(self):
        """Test parsing valid JSON payload."""
        json_str = '{"measurement_id": "G-1234567890", "client_id": "12345.67890", "events": []}'
        result = self.parser.parse_json_payload(json_str)
        
        assert "measurement_id" in result
        assert result["measurement_id"] == "G-1234567890"
        assert "client_id" in result
        assert isinstance(result["events"], list)
    
    def test_parse_json_payload_invalid(self):
        """Test parsing invalid JSON payload."""
        invalid_json = '{"invalid": json syntax}'
        result = self.parser.parse_json_payload(invalid_json)
        
        assert result == {}
    
    def test_parse_json_payload_empty(self):
        """Test parsing empty JSON payload."""
        result = self.parser.parse_json_payload("")
        assert result == {}
        
        result = self.parser.parse_json_payload(None)
        assert result == {}
    
    def test_parse_url_encoded_basic(self):
        """Test parsing basic URL-encoded parameters."""
        query = "param1=value1&param2=value2&param3="
        result = self.parser.parse_url_encoded(query)
        
        assert result["param1"] == "value1"
        assert result["param2"] == "value2"
        assert result["param3"] == ""
    
    def test_parse_url_encoded_with_encoding(self):
        """Test parsing URL-encoded parameters with special characters."""
        query = "title=Hello%20World&symbol=%26%23x2713%3B&space=test%2Bvalue"
        result = self.parser.parse_url_encoded(query)
        
        assert result["title"] == "Hello World"
        assert result["symbol"] == "&#x2713;"
        # %2B in query strings decodes to + which becomes space in parse_qs
        assert result["space"] == "test value"
    
    def test_parse_url_encoded_multiple_values(self):
        """Test parsing parameters with multiple values."""
        query = "tags=tag1&tags=tag2&tags=tag3&single=value"
        result = self.parser.parse_url_encoded(query)
        
        assert isinstance(result["tags"], list)
        assert len(result["tags"]) == 3
        assert "tag1" in result["tags"]
        assert result["single"] == "value"
    
    def test_parse_url_encoded_with_question_mark(self):
        """Test parsing query string starting with ?."""
        query = "?param1=value1&param2=value2"
        result = self.parser.parse_url_encoded(query)
        
        assert result["param1"] == "value1"
        assert result["param2"] == "value2"
    
    def test_parse_form_data_url_encoded(self):
        """Test parsing form data as URL-encoded."""
        body = "tid=G-1234567890&t=pageview&dl=https%3A%2F%2Fexample.com"
        content_type = "application/x-www-form-urlencoded"
        
        result = self.parser.parse_form_data(body, content_type)
        
        assert result["tid"] == "G-1234567890"
        assert result["t"] == "pageview"
        assert result["dl"] == "https://example.com"
    
    def test_parse_form_data_json(self):
        """Test parsing form data as JSON."""
        body = '{"measurement_id": "G-1234567890", "events": [{"name": "page_view"}]}'
        content_type = "application/json"
        
        result = self.parser.parse_form_data(body, content_type)
        
        assert result["measurement_id"] == "G-1234567890"
        assert len(result["events"]) == 1
    
    def test_extract_ga4_events_mp_format(self):
        """Test extracting GA4 events from Measurement Protocol format."""
        params = {
            "measurement_id": "G-1234567890",
            "events": [
                {
                    "name": "page_view",
                    "params": {"page_title": "Test Page"},
                    "custom_parameters": {"custom_param": "value"}
                },
                {
                    "name": "scroll",
                    "params": {"percent_scrolled": 50}
                }
            ]
        }
        
        events = self.parser.extract_ga4_events(params)
        
        assert len(events) == 2
        assert events[0]["name"] == "page_view"
        assert events[0]["params"]["page_title"] == "Test Page"
        assert events[0]["custom_params"]["custom_param"] == "value"
        assert events[1]["name"] == "scroll"
    
    def test_extract_ga4_events_single_format(self):
        """Test extracting GA4 events from single event format."""
        params = {
            "en": "button_click",
            "ep.button_text": "Subscribe",
            "ep.button_location": "header"
        }
        
        events = self.parser.extract_ga4_events(params)
        
        assert len(events) == 1
        assert events[0]["name"] == "button_click"
        assert "ep.button_text" in events[0]["params"]
    
    def test_extract_ga4_events_legacy_format(self):
        """Test extracting GA4 events from legacy gtag format."""
        params = {
            "t": "event",
            "ea": "click",
            "ec": "button", 
            "el": "header-subscribe"
        }
        
        events = self.parser.extract_ga4_events(params)
        
        assert len(events) == 1
        assert events[0]["name"] == "click"
        assert events[0]["params"]["ea"] == "click"
    
    def test_normalize_parameters(self):
        """Test parameter normalization."""
        params = {
            "TID": "G-1234567890",
            "Client_ID": "12345.67890",
            "page%20title": "Test%20Page",
            "MEASUREMENT_ID": "G-0987654321"
        }
        
        normalized = self.parser.normalize_parameters(params)
        
        # Should normalize keys and decode values
        assert "measurement_id" in normalized  # TID -> measurement_id
        assert "client_id" in normalized      # Client_ID -> client_id  
        # Key normalization only handles case/mapping, not URL decoding
        assert "page%20title" in normalized  # Key stays URL encoded
        assert normalized["page%20title"] == "Test Page"  # Value is URL decoded
    
    def test_extract_client_info(self):
        """Test extracting client information."""
        params = {
            "cid": "12345.67890",
            "sid": "session123",
            "uid": "user456",
            "other_param": "value"
        }
        
        client_info = self.parser.extract_client_info(params)
        
        assert client_info["client_id"] == "12345.67890"
        assert client_info["session_id"] == "session123"
        assert client_info["user_id"] == "user456"
        assert "other_param" not in client_info
    
    def test_extract_page_info(self):
        """Test extracting page information."""
        params = {
            "dl": "https://example.com/page",
            "dt": "Test Page Title",
            "dr": "https://referrer.com",
            "unrelated": "value"
        }
        
        page_info = self.parser.extract_page_info(params)
        
        assert page_info["page_location"] == "https://example.com/page"
        assert page_info["page_title"] == "Test Page Title" 
        assert page_info["page_referrer"] == "https://referrer.com"
        assert "unrelated" not in page_info
    
    def test_extract_custom_dimensions(self):
        """Test extracting custom dimensions and metrics."""
        params = {
            "custom_dimension_1": "value1",
            "cd2": "dimension2",
            "cm3": "metric3", 
            "custom_user_property": "prop_value",
            "regular_param": "value"
        }
        
        custom_data = self.parser.extract_custom_dimensions(params)
        
        # Custom dimensions
        assert "custom_dimension_1" in custom_data["custom_dimensions"]
        assert "custom_dimension_2" in custom_data["custom_dimensions"]
        assert "custom_user_property" in custom_data["custom_dimensions"]
        
        # Custom metrics
        assert "custom_metric_3" in custom_data["custom_metrics"]
        
        # Regular param should not be included
        assert "regular_param" not in custom_data["custom_dimensions"]
    
    def test_validate_required_params(self):
        """Test required parameter validation."""
        params = {
            "measurement_id": "G-1234567890",
            "client_id": "12345.67890",
            "empty_param": None
        }
        
        required = ["measurement_id", "client_id", "session_id", "empty_param"]
        
        result = self.parser.validate_required_params(params, required)
        
        assert result["valid"] is False
        assert "measurement_id" in result["present_params"]
        assert "client_id" in result["present_params"]
        assert "session_id" in result["missing_params"]
        assert "empty_param" in result["missing_params"]
    
    def test_sanitize_sensitive_params(self):
        """Test sensitive parameter sanitization."""
        params = {
            "measurement_id": "G-1234567890",
            "api_key": "secret123",
            "user_password": "password123",
            "access_token": "token456",
            "regular_param": "value"
        }
        
        sanitized = self.parser.sanitize_sensitive_params(params)
        
        assert sanitized["measurement_id"] == "G-1234567890"  # Not sensitive
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["user_password"] == "[REDACTED]"
        assert sanitized["access_token"] == "[REDACTED]"
        assert sanitized["regular_param"] == "value"


class TestUtilityFunctions:
    """Test standalone utility functions."""
    
    def test_extract_measurement_id_from_url(self):
        """Test extracting measurement ID from URL."""
        url = "https://www.google-analytics.com/mp/collect?measurement_id=G-1234567890"
        mid = extract_measurement_id(url)
        assert mid == "G-1234567890"
        
        # Test with no measurement ID in URL
        url_no_id = "https://www.google-analytics.com/mp/collect"
        mid_none = extract_measurement_id(url_no_id)
        assert mid_none is None
    
    def test_extract_measurement_id_from_params(self):
        """Test extracting measurement ID from parameters."""
        url = "https://www.google-analytics.com/mp/collect"
        params = {"tid": "G-9876543210"}
        
        mid = extract_measurement_id(url, params)
        assert mid == "G-9876543210"
    
    def test_extract_container_id(self):
        """Test extracting GTM container ID from URL.""" 
        url = "https://www.googletagmanager.com/gtm.js?id=GTM-ABC123"
        cid = extract_container_id(url)
        assert cid == "GTM-ABC123"
        
        # Test with no container ID
        url_no_id = "https://www.example.com/script.js"
        cid_none = extract_container_id(url_no_id)
        assert cid_none is None
    
    def test_is_analytics_request(self):
        """Test analytics request detection."""
        # GA4 requests
        assert is_analytics_request("https://www.google-analytics.com/mp/collect") is True
        assert is_analytics_request("https://www.google-analytics.com/g/collect") is True
        assert is_analytics_request("https://region1.google-analytics.com/mp/collect") is True
        
        # GTM requests
        assert is_analytics_request("https://www.googletagmanager.com/gtm.js?id=GTM-ABC") is True
        
        # Non-analytics requests
        assert is_analytics_request("https://www.example.com/api") is False
        assert is_analytics_request("https://www.facebook.com/tr") is False
    
    def test_normalize_parameter_name(self):
        """Test parameter name normalization."""
        assert normalize_parameter_name("TID") == "measurement_id"
        assert normalize_parameter_name("tracking_id") == "measurement_id"
        assert normalize_parameter_name("CID") == "client_id"
        assert normalize_parameter_name("client-id") == "client_id"
        assert normalize_parameter_name("dl") == "page_location"
        assert normalize_parameter_name("dt") == "page_title"
        
        # Unknown parameters should remain unchanged but lowercase
        assert normalize_parameter_name("UNKNOWN_PARAM") == "unknown_param"
    
    def test_validate_measurement_id(self):
        """Test measurement ID validation."""
        # Valid formats
        assert validate_measurement_id("G-1234567890") is True
        assert validate_measurement_id("G-ABCDEFGHIJ") is True
        
        # Invalid formats
        assert validate_measurement_id("GA-1234567890") is False  # Wrong prefix
        assert validate_measurement_id("G-123") is False         # Too short
        assert validate_measurement_id("1234567890") is False    # No prefix
        assert validate_measurement_id("") is False              # Empty
        assert validate_measurement_id(None) is False            # None
    
    def test_validate_container_id(self):
        """Test container ID validation."""
        # Valid formats
        assert validate_container_id("GTM-ABC123") is True
        assert validate_container_id("GTM-1234567") is True
        assert validate_container_id("GTM-ABCDEFG") is True
        
        # Invalid formats
        assert validate_container_id("GM-ABC123") is False   # Wrong prefix
        assert validate_container_id("GTM-AB") is False      # Too short  
        assert validate_container_id("ABC123") is False      # No prefix
        assert validate_container_id("") is False            # Empty
        assert validate_container_id(None) is False          # None


class TestDefaultPatterns:
    """Test default analytics patterns."""
    
    def test_default_patterns_loaded(self):
        """Test that default patterns are loaded."""
        # GA4 patterns
        assert patterns.get_pattern("ga4_mp_collect") is not None
        assert patterns.get_pattern("ga4_g_collect") is not None
        assert patterns.get_pattern("ga4_measurement_id") is not None
        
        # GTM patterns
        assert patterns.get_pattern("gtm_loader") is not None
        assert patterns.get_pattern("gtm_container_id") is not None
    
    def test_ga4_patterns(self):
        """Test GA4 pattern matching."""
        mp_pattern = patterns.get_pattern("ga4_mp_collect")
        assert mp_pattern.search("https://www.google-analytics.com/mp/collect") is not None
        assert mp_pattern.search("https://region1.google-analytics.com/mp/collect") is not None
        
        measurement_id_pattern = patterns.get_pattern("ga4_measurement_id")
        assert measurement_id_pattern.match("G-1234567890") is not None
        assert measurement_id_pattern.match("G-ABCDEFGHIJ") is not None
        assert measurement_id_pattern.match("GA-1234567890") is None
    
    def test_gtm_patterns(self):
        """Test GTM pattern matching."""
        loader_pattern = patterns.get_pattern("gtm_loader")
        match = loader_pattern.search("https://www.googletagmanager.com/gtm.js?id=GTM-ABC123")
        assert match is not None
        assert match.group(1) == "GTM-ABC123"  # Captured container ID
        
        container_id_pattern = patterns.get_pattern("gtm_container_id")
        assert container_id_pattern.match("GTM-ABC123") is not None
        assert container_id_pattern.match("GTM-1234567") is not None
        assert container_id_pattern.match("GM-ABC123") is None