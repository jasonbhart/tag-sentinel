"""Scope matching engine for URL filtering during crawl operations.

This module implements include/exclude pattern matching with proper precedence
handling and same-site restriction capabilities.
"""

import re
from typing import List, Optional, Set, Tuple
from urllib.parse import urlparse
import threading

from .url_normalizer import normalize, are_same_site, URLNormalizationError


class ScopeMatcherError(Exception):
    """Raised when scope matcher encounters an error."""
    pass


class ScopeMatcher:
    """Thread-safe scope matcher for URL filtering with include/exclude patterns.
    
    The matcher applies the following logic:
    1. URLs must match at least one include pattern (if any are specified)
    2. URLs must not match any exclude patterns
    3. If same_site_only is True, URLs must be same-site as reference URLs
    
    Exclude patterns always take precedence over include patterns.
    """
    
    def __init__(
        self,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        same_site_only: bool = False,
        reference_urls: Optional[List[str]] = None
    ):
        """Initialize the scope matcher.
        
        Args:
            include_patterns: Regex patterns for URLs to include
            exclude_patterns: Regex patterns for URLs to exclude  
            same_site_only: Whether to restrict to same site as reference URLs
            reference_urls: Reference URLs for same-site filtering (e.g., seed URLs)
            
        Raises:
            ScopeMatcherError: If regex patterns are invalid
        """
        self._lock = threading.RLock()
        
        # Compile include patterns
        self._include_patterns = []
        if include_patterns:
            for pattern in include_patterns:
                try:
                    compiled = re.compile(pattern)
                    self._include_patterns.append((pattern, compiled))
                except re.error as e:
                    raise ScopeMatcherError(f"Invalid include pattern '{pattern}': {e}")
        
        # Compile exclude patterns
        self._exclude_patterns = []
        if exclude_patterns:
            for pattern in exclude_patterns:
                try:
                    compiled = re.compile(pattern)
                    self._exclude_patterns.append((pattern, compiled))
                except re.error as e:
                    raise ScopeMatcherError(f"Invalid exclude pattern '{pattern}': {e}")
        
        # Same-site configuration
        self._same_site_only = same_site_only
        self._reference_urls = set()
        if same_site_only and reference_urls:
            for url in reference_urls:
                try:
                    normalized = normalize(url)
                    self._reference_urls.add(normalized)
                except URLNormalizationError:
                    # Skip invalid reference URLs but don't fail initialization
                    continue
    
    def is_in_scope(self, url: str) -> bool:
        """Check if a URL is within the configured scope.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL is in scope, False otherwise
        """
        try:
            # Normalize URL for consistent matching
            normalized_url = normalize(url)
        except URLNormalizationError:
            # Invalid URLs are always out of scope
            return False
        
        with self._lock:
            # Check same-site restriction first (most restrictive)
            if self._same_site_only and self._reference_urls:
                if not self._is_same_site(normalized_url):
                    return False
            
            # Check exclude patterns (highest precedence)
            for pattern_str, pattern in self._exclude_patterns:
                if pattern.search(normalized_url):
                    return False
            
            # Check include patterns (if any exist)
            if self._include_patterns:
                for pattern_str, pattern in self._include_patterns:
                    if pattern.search(normalized_url):
                        return True
                # If include patterns exist but none match, URL is out of scope
                return False
            
            # No include patterns specified and not excluded = in scope
            return True
    
    def filter_urls(self, urls: List[str]) -> Tuple[List[str], List[str]]:
        """Filter a list of URLs into in-scope and out-of-scope lists.
        
        Args:
            urls: List of URLs to filter
            
        Returns:
            Tuple of (in_scope_urls, out_of_scope_urls)
        """
        in_scope = []
        out_of_scope = []
        
        for url in urls:
            if self.is_in_scope(url):
                in_scope.append(url)
            else:
                out_of_scope.append(url)
        
        return in_scope, out_of_scope
    
    def get_scope_info(self) -> dict:
        """Get information about the configured scope.
        
        Returns:
            Dictionary with scope configuration details
        """
        with self._lock:
            return {
                "include_patterns": [pattern for pattern, _ in self._include_patterns],
                "exclude_patterns": [pattern for pattern, _ in self._exclude_patterns],
                "same_site_only": self._same_site_only,
                "reference_urls": len(self._reference_urls),
                "has_includes": len(self._include_patterns) > 0,
                "has_excludes": len(self._exclude_patterns) > 0
            }
    
    def add_reference_url(self, url: str) -> bool:
        """Add a reference URL for same-site filtering.
        
        Args:
            url: URL to add as reference for same-site filtering
            
        Returns:
            True if URL was added successfully, False if invalid
        """
        try:
            normalized = normalize(url)
            with self._lock:
                self._reference_urls.add(normalized)
                return True
        except URLNormalizationError:
            return False
    
    def _is_same_site(self, url: str) -> bool:
        """Check if URL is same-site as any reference URL.
        
        Args:
            url: Normalized URL to check
            
        Returns:
            True if same-site as any reference URL, False otherwise
        """
        if not self._reference_urls:
            return True  # No reference URLs = no restriction
        
        for ref_url in self._reference_urls:
            if are_same_site(url, ref_url):
                return True
        
        return False
    
    def get_match_details(self, url: str) -> dict:
        """Get detailed information about why a URL matched or didn't match.
        
        This is useful for debugging scope configuration.
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dictionary with detailed matching information
        """
        try:
            normalized_url = normalize(url)
        except URLNormalizationError as e:
            return {
                "url": url,
                "normalized": None,
                "in_scope": False,
                "reason": "invalid_url",
                "error": str(e),
                "checks": []
            }
        
        checks = []
        in_scope = True
        reason = "allowed"
        
        with self._lock:
            # Same-site check
            if self._same_site_only and self._reference_urls:
                is_same_site = self._is_same_site(normalized_url)
                checks.append({
                    "check": "same_site",
                    "required": True,
                    "passed": is_same_site,
                    "reference_count": len(self._reference_urls)
                })
                if not is_same_site:
                    in_scope = False
                    reason = "different_site"
            
            # Exclude pattern check
            excluded_by = None
            for pattern_str, pattern in self._exclude_patterns:
                matches = pattern.search(normalized_url) is not None
                checks.append({
                    "check": "exclude",
                    "pattern": pattern_str,
                    "matches": matches
                })
                if matches:
                    in_scope = False
                    reason = "excluded"
                    excluded_by = pattern_str
                    break
            
            # Include pattern check (only if not already excluded)
            if in_scope and self._include_patterns:
                any_include_match = False
                for pattern_str, pattern in self._include_patterns:
                    matches = pattern.search(normalized_url) is not None
                    checks.append({
                        "check": "include", 
                        "pattern": pattern_str,
                        "matches": matches
                    })
                    if matches:
                        any_include_match = True
                
                if not any_include_match:
                    in_scope = False
                    reason = "no_include_match"
        
        return {
            "url": url,
            "normalized": normalized_url,
            "in_scope": in_scope,
            "reason": reason,
            "excluded_by": excluded_by,
            "checks": checks
        }


def create_scope_matcher_from_config(config) -> ScopeMatcher:
    """Create a ScopeMatcher from a crawl configuration object.
    
    Args:
        config: CrawlConfig object with scope settings
        
    Returns:
        Configured ScopeMatcher instance
    """
    reference_urls = []
    if config.seeds:
        reference_urls = [str(url) for url in config.seeds]
    
    return ScopeMatcher(
        include_patterns=config.include_patterns or None,
        exclude_patterns=config.exclude_patterns or None,
        same_site_only=config.same_site_only,
        reference_urls=reference_urls
    )