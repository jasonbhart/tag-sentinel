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


if __name__ == "__main__":
    pytest.main([__file__])