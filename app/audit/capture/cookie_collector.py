"""Cookie collector for privacy-conscious cookie analysis.

This module provides the CookieCollector class that extracts and analyzes
cookies from browser contexts, with support for privacy-conscious handling,
first-party vs third-party classification, and detailed attribute analysis.
"""

import logging
from typing import List, Dict, Set, Optional, Callable
from urllib.parse import urlparse

from playwright.async_api import BrowserContext

from ..models.capture import CookieRecord

logger = logging.getLogger(__name__)


class CookieCollector:
    """Collector for analyzing cookies with privacy-conscious handling."""
    
    def __init__(self, context: BrowserContext, redact_values: bool = True):
        """Initialize cookie collector.
        
        Args:
            context: Playwright browser context
            redact_values: Whether to redact cookie values for privacy
        """
        self.context = context
        self.redact_values = redact_values
        self.cookies: List[CookieRecord] = []
        self._callbacks: List[Callable[[List[CookieRecord]], None]] = []
        
        # Known analytics and advertising domains for classification
        self._analytics_domains = {
            'google-analytics.com', 'googletagmanager.com', 'google.com', 'doubleclick.net',
            'facebook.com', 'facebook.net', 'connect.facebook.net',
            'adobe.com', 'omtrdc.net', 'demdex.net', 'everesttech.net',
            'hotjar.com', 'hotjar.io',
            'mixpanel.com', 'mixpanel.org', 
            'segment.com', 'segment.io',
            'fullstory.com', 'fullstory.org',
            'amplitude.com', 'amplify.com',
            'crazyegg.com', 'clicktale.com',
            'quantserve.com', 'scorecardresearch.com',
            'amazon-adsystem.com', 'adsystem.amazon.com',
            'bing.com', 'bing.net', 'live.com',
            'twitter.com', 'twimg.com',
            'linkedin.com', 'licdn.com',
            'pinterest.com', 'pinimg.com',
            'snapchat.com', 'sc-cdn.net',
            'tiktok.com', 'musical.ly',
        }
        
        # Common cookie name patterns for analytics/tracking
        self._tracking_cookie_patterns = {
            '_ga', '_gid', '_gat', '__utma', '__utmb', '__utmc', '__utmz', '__utmv',
            'fbp', 'fbc', '_fbp', '_fbc',
            's_', 'mbox', 'AMCV_', 'AMCVS_',
            '_hjid', '_hjFirstSeen', '_hjIncludedInSessionSample',
            'mp_', '_mixpanel',
            '_segment_', 'ajs_',
            '_vwo_', 'optimizelyEndUserId',
            '_ceg', 'ct_', 'ClickTale',
            '__qca', '_quantserve',
            'ad-id', 'uuid', 'id', 'uid',
            'test_cookie', 'IDE', 'DSID', 'FLC',
            'fr', 'datr', 'sb', 'c_user',
        }
    
    def add_callback(self, callback: Callable[[List[CookieRecord]], None]) -> None:
        """Add callback to be called when cookies are collected.
        
        Args:
            callback: Function to call with cookie list
        """
        self._callbacks.append(callback)
    
    def _is_analytics_domain(self, domain: str) -> bool:
        """Check if domain is a known analytics/advertising domain.
        
        Args:
            domain: Domain to check
            
        Returns:
            True if domain is analytics/advertising related
        """
        clean_domain = domain.lstrip('.')
        
        # Direct match
        if clean_domain in self._analytics_domains:
            return True
        
        # Check if it's a subdomain of known analytics domain
        for analytics_domain in self._analytics_domains:
            if clean_domain.endswith(f'.{analytics_domain}'):
                return True
        
        return False
    
    def _is_tracking_cookie(self, cookie_name: str) -> bool:
        """Check if cookie name indicates tracking/analytics purpose.
        
        Args:
            cookie_name: Cookie name to check
            
        Returns:
            True if cookie appears to be for tracking
        """
        # Check exact matches
        if cookie_name in self._tracking_cookie_patterns:
            return True
        
        # Check prefix matches
        for pattern in self._tracking_cookie_patterns:
            if cookie_name.startswith(pattern):
                return True
        
        return False
    
    def _classify_cookie_purpose(self, cookie: dict, page_host: str) -> str:
        """Classify cookie purpose based on domain and name.
        
        Args:
            cookie: Cookie dictionary from Playwright
            page_host: Host of the page that set the cookie
            
        Returns:
            Cookie purpose classification string
        """
        cookie_domain = cookie.get('domain', '').lstrip('.')
        cookie_name = cookie.get('name', '')
        
        # Check if it's from analytics domain
        if self._is_analytics_domain(cookie_domain):
            return 'analytics'
        
        # Check if cookie name indicates tracking
        if self._is_tracking_cookie(cookie_name):
            return 'tracking'
        
        # Check if it's a session cookie
        if cookie.get('expires', -1) == -1:
            return 'session'
        
        # Check for authentication patterns
        auth_patterns = ['auth', 'token', 'jwt', 'session', 'login', 'user']
        if any(pattern in cookie_name.lower() for pattern in auth_patterns):
            return 'authentication'
        
        # Check for preference/settings patterns
        pref_patterns = ['pref', 'setting', 'config', 'theme', 'lang', 'locale', 'timezone']
        if any(pattern in cookie_name.lower() for pattern in pref_patterns):
            return 'preferences'
        
        # Check for functionality patterns
        func_patterns = ['cart', 'basket', 'wishlist', 'favorite', 'bookmark']
        if any(pattern in cookie_name.lower() for pattern in func_patterns):
            return 'functionality'
        
        # Check if first-party
        is_first_party = (
            cookie_domain == page_host or
            page_host.endswith(f'.{cookie_domain}') or
            cookie_domain.endswith(f'.{page_host}')
        )
        
        return 'necessary' if is_first_party else 'unknown'
    
    async def collect_cookies(self, page_url: str) -> List[CookieRecord]:
        """Collect and analyze cookies from browser context.
        
        Args:
            page_url: URL of the page (for first-party classification)
            
        Returns:
            List of CookieRecord objects
        """
        try:
            page_host = urlparse(page_url).netloc
            
            # Get cookies from context
            playwright_cookies = await self.context.cookies()
            
            self.cookies = []
            
            for pw_cookie in playwright_cookies:
                try:
                    # Create CookieRecord with privacy handling
                    cookie_record = CookieRecord.from_playwright_cookie(
                        pw_cookie, 
                        page_host, 
                        redact_value=self.redact_values
                    )
                    
                    # Add purpose classification
                    cookie_record.metadata = {
                        'purpose': self._classify_cookie_purpose(pw_cookie, page_host),
                        'is_analytics': self._is_analytics_domain(cookie_record.domain),
                        'is_tracking': self._is_tracking_cookie(cookie_record.name),
                    }
                    
                    self.cookies.append(cookie_record)
                    
                except Exception as e:
                    logger.warning(f"Failed to process cookie {pw_cookie.get('name', 'unknown')}: {e}")
            
            # Call callbacks
            for callback in self._callbacks:
                try:
                    callback(self.cookies)
                except Exception as e:
                    logger.error(f"Error in cookie callback: {e}")
            
            logger.info(f"Collected {len(self.cookies)} cookies for {page_url}")
            return self.cookies.copy()
            
        except Exception as e:
            logger.error(f"Failed to collect cookies: {e}")
            return []
    
    def get_cookies(self) -> List[CookieRecord]:
        """Get all collected cookies.
        
        Returns:
            List of CookieRecord objects
        """
        return self.cookies.copy()
    
    def get_first_party_cookies(self) -> List[CookieRecord]:
        """Get only first-party cookies.
        
        Returns:
            List of first-party CookieRecord objects
        """
        return [cookie for cookie in self.cookies if cookie.is_first_party]
    
    def get_third_party_cookies(self) -> List[CookieRecord]:
        """Get only third-party cookies.
        
        Returns:
            List of third-party CookieRecord objects
        """
        return [cookie for cookie in self.cookies if not cookie.is_first_party]
    
    def get_session_cookies(self) -> List[CookieRecord]:
        """Get only session cookies.
        
        Returns:
            List of session CookieRecord objects
        """
        return [cookie for cookie in self.cookies if cookie.is_session]
    
    def get_persistent_cookies(self) -> List[CookieRecord]:
        """Get only persistent cookies.
        
        Returns:
            List of persistent CookieRecord objects
        """
        return [cookie for cookie in self.cookies if not cookie.is_session]
    
    def get_cookies_by_domain(self, domain: str) -> List[CookieRecord]:
        """Get cookies for a specific domain.
        
        Args:
            domain: Domain to filter by
            
        Returns:
            List of CookieRecord objects for the domain
        """
        return [cookie for cookie in self.cookies if domain in cookie.domain]
    
    def get_cookies_by_purpose(self, purpose: str) -> List[CookieRecord]:
        """Get cookies by classified purpose.
        
        Args:
            purpose: Purpose to filter by (analytics, tracking, session, etc.)
            
        Returns:
            List of CookieRecord objects with specified purpose
        """
        return [
            cookie for cookie in self.cookies 
            if cookie.metadata and cookie.metadata.get('purpose') == purpose
        ]
    
    def get_analytics_cookies(self) -> List[CookieRecord]:
        """Get cookies from known analytics domains.
        
        Returns:
            List of analytics CookieRecord objects
        """
        return [
            cookie for cookie in self.cookies
            if cookie.metadata and cookie.metadata.get('is_analytics', False)
        ]
    
    def get_tracking_cookies(self) -> List[CookieRecord]:
        """Get cookies with tracking/advertising names.
        
        Returns:
            List of tracking CookieRecord objects
        """
        return [
            cookie for cookie in self.cookies
            if cookie.metadata and cookie.metadata.get('is_tracking', False)
        ]
    
    def get_unique_domains(self) -> Set[str]:
        """Get unique domains that set cookies.
        
        Returns:
            Set of unique domain strings
        """
        return {cookie.domain for cookie in self.cookies}
    
    def get_cookie_size_by_domain(self) -> Dict[str, int]:
        """Get total cookie size per domain.
        
        Returns:
            Dictionary mapping domain to total cookie size
        """
        domain_sizes = {}
        for cookie in self.cookies:
            domain = cookie.domain
            domain_sizes[domain] = domain_sizes.get(domain, 0) + cookie.size
        return domain_sizes
    
    def get_stats(self) -> Dict[str, any]:
        """Get comprehensive cookie statistics.
        
        Returns:
            Dictionary with detailed cookie statistics
        """
        first_party = self.get_first_party_cookies()
        third_party = self.get_third_party_cookies()
        session_cookies = self.get_session_cookies()
        persistent_cookies = self.get_persistent_cookies()
        analytics_cookies = self.get_analytics_cookies()
        tracking_cookies = self.get_tracking_cookies()
        
        total_size = sum(cookie.size for cookie in self.cookies)
        domain_sizes = self.get_cookie_size_by_domain()
        
        return {
            'total_cookies': len(self.cookies),
            'first_party_cookies': len(first_party),
            'third_party_cookies': len(third_party),
            'session_cookies': len(session_cookies),
            'persistent_cookies': len(persistent_cookies),
            'analytics_cookies': len(analytics_cookies),
            'tracking_cookies': len(tracking_cookies),
            'unique_domains': len(self.get_unique_domains()),
            'total_size_bytes': total_size,
            'average_cookie_size': total_size / len(self.cookies) if self.cookies else 0,
            'largest_domain_size': max(domain_sizes.values()) if domain_sizes else 0,
            'domains_with_cookies': list(self.get_unique_domains()),
        }
    
    def export_privacy_report(self) -> Dict[str, any]:
        """Export privacy-focused cookie report.
        
        Returns:
            Dictionary with privacy analysis
        """
        stats = self.get_stats()
        
        # Analyze security attributes
        secure_cookies = len([c for c in self.cookies if c.secure])
        http_only_cookies = len([c for c in self.cookies if c.http_only])
        same_site_cookies = len([c for c in self.cookies if c.same_site])
        
        # Purpose breakdown
        purposes = {}
        for cookie in self.cookies:
            purpose = cookie.metadata.get('purpose', 'unknown') if cookie.metadata else 'unknown'
            purposes[purpose] = purposes.get(purpose, 0) + 1
        
        return {
            'cookie_count': stats['total_cookies'],
            'third_party_cookies': stats['third_party_cookies'],
            'analytics_cookies': stats['analytics_cookies'],
            'tracking_cookies': stats['tracking_cookies'],
            'security_attributes': {
                'secure_cookies': secure_cookies,
                'http_only_cookies': http_only_cookies,
                'same_site_cookies': same_site_cookies,
                'secure_percentage': (secure_cookies / len(self.cookies) * 100) if self.cookies else 0,
            },
            'cookie_purposes': purposes,
            'privacy_impact': {
                'high': stats['tracking_cookies'] + stats['analytics_cookies'],
                'medium': stats['third_party_cookies'] - stats['tracking_cookies'] - stats['analytics_cookies'],
                'low': stats['first_party_cookies'],
            },
            'redaction_enabled': self.redact_values,
        }
    
    def clear(self) -> None:
        """Clear all collected cookies."""
        self.cookies.clear()
        logger.debug("Cookie collector cleared")
    
    def __repr__(self) -> str:
        """String representation of cookie collector."""
        stats = self.get_stats()
        return (
            f"CookieCollector(total={stats['total_cookies']}, "
            f"first_party={stats['first_party_cookies']}, "
            f"third_party={stats['third_party_cookies']}, "
            f"domains={stats['unique_domains']})"
        )