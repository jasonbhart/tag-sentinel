"""Unit tests for URL normalization utilities."""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.audit.utils.url_normalizer import (
    normalize,
    are_same_site,
    get_base_url,
    is_valid_http_url,
    URLNormalizationError
)


class TestURLNormalizer:
    """Test cases for URL normalization."""
    
    def test_basic_normalization(self):
        """Test basic URL normalization cases."""
        test_cases = [
            ("HTTP://Example.COM:80/Path/?param=value#fragment", "http://example.com/Path/?param=value"),
            ("https://site.com:443/", "https://site.com/"),
            ("http://site.com/path#fragment", "http://site.com/path"),
            ("https://example.com", "https://example.com/"),  # Adds trailing slash
        ]
        
        for input_url, expected in test_cases:
            result = normalize(input_url)
            assert result == expected, f"Expected {expected}, got {result}"
    
    def test_same_site_comparison(self):
        """Test same-site URL comparison."""
        assert are_same_site("https://www.example.com/page1", "https://blog.example.com/page2")
        assert not are_same_site("https://example.com/page1", "https://other.com/page2")
        assert are_same_site("https://example.com/page1", "https://example.com/page2")
    
    def test_url_validation(self):
        """Test URL validation."""
        assert is_valid_http_url("https://example.com")
        assert is_valid_http_url("http://example.com/path")
        assert not is_valid_http_url("ftp://example.com")
        assert not is_valid_http_url("not-a-url")
        assert not is_valid_http_url("")
    
    def test_base_url_extraction(self):
        """Test base URL extraction."""
        assert get_base_url("https://example.com/path/to/page?param=value") == "https://example.com"
        assert get_base_url("http://sub.example.com:8080/page") == "http://sub.example.com:8080"
    
    def test_normalization_errors(self):
        """Test URL normalization error handling."""
        with pytest.raises(URLNormalizationError):
            normalize("")

        with pytest.raises(URLNormalizationError):
            normalize("not-a-url")

        with pytest.raises(URLNormalizationError):
            normalize("ftp://example.com")

    def test_ipv6_normalization(self):
        """Test IPv6 URL normalization."""
        test_cases = [
            # IPv6 without port
            ("http://[2001:db8::1]/path", "http://[2001:db8::1]/path"),
            ("https://[2001:db8::1]/path", "https://[2001:db8::1]/path"),

            # IPv6 with default ports (should be removed)
            ("http://[2001:db8::1]:80/path", "http://[2001:db8::1]/path"),
            ("https://[2001:db8::1]:443/path", "https://[2001:db8::1]/path"),

            # IPv6 with non-default ports (should be kept)
            ("http://[2001:db8::1]:8080/path", "http://[2001:db8::1]:8080/path"),
            ("https://[2001:db8::1]:8443/path", "https://[2001:db8::1]:8443/path"),

            # IPv6 case normalization
            ("http://[2001:DB8::1]/path", "http://[2001:db8::1]/path"),
        ]

        for input_url, expected in test_cases:
            result = normalize(input_url)
            assert result == expected, f"Expected {expected}, got {result}"

    def test_idn_normalization(self):
        """Test internationalized domain name (IDN) normalization."""
        test_cases = [
            # Common IDN cases
            ("http://例え.テスト/path", "http://xn--r8jz45g.xn--zckzah/path"),
            ("https://münchen.de/path", "https://xn--mnchen-3ya.de/path"),
            ("http://пример.испытание/path", "http://xn--e1afmkfd.xn--80akhbyknj4f/path"),
        ]

        for input_url, expected in test_cases:
            result = normalize(input_url)
            assert result == expected, f"Expected {expected}, got {result}"

    def test_mixed_ipv6_and_edge_cases(self):
        """Test edge cases and mixed scenarios."""
        test_cases = [
            # IPv6 with userinfo (rare but possible)
            ("http://user:pass@[2001:db8::1]:8080/path", "http://user:pass@[2001:db8::1]:8080/path"),

            # Malformed IPv6 (should be handled gracefully)
            ("http://[2001:db8::1/path", "http://[2001:db8::1/path"),  # Missing closing bracket

            # Regular hostname with colon but no port
            ("http://example.com:/path", "http://example.com:/path"),  # Invalid but handled
        ]

        for input_url, expected in test_cases:
            try:
                result = normalize(input_url)
                assert result == expected, f"Expected {expected}, got {result}"
            except URLNormalizationError:
                # Some malformed URLs should raise errors, which is acceptable
                pass

    def test_port_handling_edge_cases(self):
        """Test edge cases for port handling."""
        test_cases = [
            # Valid port numbers
            ("http://example.com:8080/path", "http://example.com:8080/path"),
            ("https://example.com:9443/path", "https://example.com:9443/path"),

            # Invalid port numbers (non-numeric)
            ("http://example.com:abc/path", "http://example.com:abc/path"),  # Should be handled gracefully

            # Edge case: port but no number
            ("http://example.com:/path", "http://example.com:/path"),  # Should be handled gracefully
        ]

        for input_url, expected in test_cases:
            try:
                result = normalize(input_url)
                # For invalid ports, we just keep them as-is (graceful handling)
                assert result == expected, f"Expected {expected}, got {result}"
            except URLNormalizationError:
                # Some cases might raise errors, which is acceptable
                pass


if __name__ == "__main__":
    pytest.main([__file__])