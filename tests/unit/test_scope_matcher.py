"""Unit tests for scope matching functionality."""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.audit.utils.scope_matcher import ScopeMatcher, ScopeMatcherError


class TestScopeMatcher:
    """Test cases for scope matching."""
    
    def test_include_patterns(self):
        """Test include pattern matching."""
        matcher = ScopeMatcher(
            include_patterns=[".*example\\.com.*"],
            same_site_only=False
        )
        
        assert matcher.is_in_scope("https://example.com/page")
        assert matcher.is_in_scope("https://sub.example.com/page")
        assert not matcher.is_in_scope("https://other.com/page")
    
    def test_exclude_patterns(self):
        """Test exclude pattern matching."""
        matcher = ScopeMatcher(
            include_patterns=[".*example\\.com.*"],
            exclude_patterns=[".*/admin/.*"],
            same_site_only=False
        )
        
        assert matcher.is_in_scope("https://example.com/page")
        assert not matcher.is_in_scope("https://example.com/admin/panel")
    
    def test_same_site_restriction(self):
        """Test same-site only restriction."""
        matcher = ScopeMatcher(
            same_site_only=True,
            reference_urls=["https://example.com"]
        )
        
        assert matcher.is_in_scope("https://example.com/page")
        assert matcher.is_in_scope("https://sub.example.com/page")
        assert not matcher.is_in_scope("https://other.com/page")
    
    def test_filter_urls(self):
        """Test bulk URL filtering."""
        matcher = ScopeMatcher(
            include_patterns=[".*example\\.com.*"],
            exclude_patterns=[".*/admin/.*"],
            same_site_only=False
        )
        
        urls = [
            "https://example.com/page",
            "https://example.com/admin/panel",
            "https://other.com/page"
        ]
        
        in_scope, out_of_scope = matcher.filter_urls(urls)
        
        assert len(in_scope) == 1
        assert "https://example.com/page" in in_scope
        assert len(out_of_scope) == 2
    
    def test_invalid_regex_patterns(self):
        """Test invalid regex pattern handling."""
        with pytest.raises(ScopeMatcherError):
            ScopeMatcher(include_patterns=["[invalid regex"])
    
    def test_match_details(self):
        """Test detailed match information."""
        matcher = ScopeMatcher(
            include_patterns=[".*example\\.com.*"],
            exclude_patterns=[".*/admin/.*"],
            same_site_only=False
        )
        
        details = matcher.get_match_details("https://example.com/page")
        assert details["in_scope"] is True
        assert details["reason"] == "allowed"
        
        details = matcher.get_match_details("https://example.com/admin/panel")
        assert details["in_scope"] is False
        assert details["reason"] == "excluded"


if __name__ == "__main__":
    pytest.main([__file__])