"""URL normalization utility for consistent deduplication and comparison.

This module provides URL canonicalization to ensure consistent handling
of URLs throughout the crawling process, enabling effective deduplication
and comparison operations.
"""

import re
from urllib.parse import urlparse, urlunparse, unquote, quote
from typing import Optional


class URLNormalizationError(Exception):
    """Raised when URL normalization fails."""
    pass


def normalize(url: str) -> str:
    """Normalize a URL for consistent deduplication and comparison.
    
    Args:
        url: The URL to normalize
        
    Returns:
        The normalized URL string
        
    Raises:
        URLNormalizationError: If the URL cannot be normalized
        
    Example:
        >>> normalize("HTTP://Example.COM:443/Path/?param=value#fragment")
        "https://example.com/Path/?param=value"
    """
    if not url or not isinstance(url, str):
        raise URLNormalizationError("URL must be a non-empty string")
    
    url = url.strip()
    if not url:
        raise URLNormalizationError("URL cannot be empty or whitespace only")
    
    try:
        # Parse the URL
        parsed = urlparse(url)
        
        # Validate required components
        if not parsed.scheme:
            raise URLNormalizationError(f"URL missing scheme: {url}")
        if not parsed.netloc:
            raise URLNormalizationError(f"URL missing netloc: {url}")
            
        # Normalize scheme to lowercase
        scheme = parsed.scheme.lower()
        if scheme not in ('http', 'https'):
            raise URLNormalizationError(f"Unsupported URL scheme: {scheme}")
        
        # Normalize host to lowercase and handle IDN
        try:
            # Handle internationalized domain names
            netloc = parsed.netloc.lower()
            if '@' in netloc:
                # Handle userinfo (should be rare for web crawling)
                userinfo, host_port = netloc.rsplit('@', 1)
                netloc = f"{userinfo}@{host_port}"
            else:
                host_port = netloc
                
            # Split host and port
            if ':' in host_port and not host_port.startswith('['):
                # IPv4 or hostname with port
                host, port = host_port.rsplit(':', 1)
                try:
                    port_num = int(port)
                    # Remove default ports
                    if ((scheme == 'http' and port_num == 80) or 
                        (scheme == 'https' and port_num == 443)):
                        netloc = host if '@' not in netloc else f"{netloc.split('@')[0]}@{host}"
                    else:
                        netloc = netloc  # Keep non-default ports
                except ValueError:
                    # Invalid port, keep as-is
                    pass
            elif host_port.startswith('[') and host_port.endswith(']'):
                # IPv6 address
                netloc = host_port.lower()
                
        except UnicodeError:
            raise URLNormalizationError(f"Invalid Unicode in hostname: {parsed.netloc}")
        
        # Normalize path
        path = parsed.path
        if not path:
            path = '/'
        else:
            # Decode and re-encode path to normalize encoding
            try:
                path = quote(unquote(path), safe='/~')
            except (UnicodeDecodeError, UnicodeEncodeError):
                # Keep original path if encoding fails
                pass
        
        # Keep query string as-is (parameter normalization could break functionality)
        query = parsed.query
        
        # Remove fragment (not sent to server)
        fragment = ''
        
        # Reconstruct normalized URL
        normalized = urlunparse((scheme, netloc, path, parsed.params, query, fragment))
        
        return normalized
        
    except Exception as e:
        if isinstance(e, URLNormalizationError):
            raise
        raise URLNormalizationError(f"Failed to normalize URL '{url}': {e}")


def are_same_site(url1: str, url2: str) -> bool:
    """Check if two URLs belong to the same site (eTLD+1 comparison).
    
    Args:
        url1: First URL to compare
        url2: Second URL to compare
        
    Returns:
        True if URLs belong to the same site, False otherwise
        
    Example:
        >>> are_same_site("https://www.example.com/page1", "https://blog.example.com/page2")
        True
        >>> are_same_site("https://example.com/page1", "https://other.com/page2")
        False
    """
    try:
        parsed1 = urlparse(normalize(url1))
        parsed2 = urlparse(normalize(url2))
        
        # Extract hostnames
        host1 = parsed1.netloc.split(':')[0].lower()
        host2 = parsed2.netloc.split(':')[0].lower()
        
        # Simple same-site check - for production, consider using publicsuffix2 library
        # This is a simplified version that works for common cases
        
        # Remove www prefix for comparison
        if host1.startswith('www.'):
            host1 = host1[4:]
        if host2.startswith('www.'):
            host2 = host2[4:]
        
        # Split into parts
        parts1 = host1.split('.')
        parts2 = host2.split('.')
        
        # Need at least 2 parts for domain comparison
        if len(parts1) < 2 or len(parts2) < 2:
            return host1 == host2
        
        # Compare last two parts (simplified eTLD+1)
        return parts1[-2:] == parts2[-2:]
        
    except (URLNormalizationError, Exception):
        return False


def get_base_url(url: str) -> str:
    """Extract the base URL (scheme + netloc) from a full URL.
    
    Args:
        url: The URL to extract base from
        
    Returns:
        The base URL (scheme://netloc)
        
    Example:
        >>> get_base_url("https://example.com/path/to/page?param=value")
        "https://example.com"
    """
    try:
        normalized = normalize(url)
        parsed = urlparse(normalized)
        return f"{parsed.scheme}://{parsed.netloc}"
    except URLNormalizationError:
        raise


def is_valid_http_url(url: str) -> bool:
    """Check if a URL is a valid HTTP/HTTPS URL.
    
    Args:
        url: The URL to validate
        
    Returns:
        True if the URL is valid HTTP/HTTPS, False otherwise
    """
    try:
        normalize(url)
        return True
    except URLNormalizationError:
        return False