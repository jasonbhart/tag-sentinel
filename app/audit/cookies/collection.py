"""Enhanced cookie collection for privacy analysis.

This module extends the base cookie collection functionality with privacy-specific
enhancements including temporal tracking, enhanced classification, and scenario-aware
collection for Epic 5 - Cookies & Consent.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Callable, Any
from urllib.parse import urlparse

from playwright.async_api import BrowserContext, Page

from ..models.capture import CookieRecord as BaseCookieRecord
from .models import CookieRecord
from .config import PrivacyConfiguration

logger = logging.getLogger(__name__)


class EnhancedCookieCollector:
    """Enhanced cookie collector with privacy-specific functionality.
    
    Extends the base cookie collection with temporal tracking, enhanced
    classification, and scenario-aware collection capabilities.
    """
    
    def __init__(
        self, 
        context: BrowserContext, 
        config: Optional[PrivacyConfiguration] = None,
        redact_values: bool = True
    ):
        """Initialize enhanced cookie collector.
        
        Args:
            context: Playwright browser context
            config: Privacy configuration for classification rules
            redact_values: Whether to redact cookie values for privacy
        """
        self.context = context
        self.config = config
        self.redact_values = redact_values
        
        # Cookie tracking
        self.cookies: List[CookieRecord] = []
        self.cookie_timeline: Dict[str, List[datetime]] = {}  # cookie_key -> timestamps
        self.initial_cookies: Optional[List[CookieRecord]] = None
        
        # Callbacks for real-time processing
        self._callbacks: List[Callable[[List[CookieRecord]], None]] = []
        
        # Collection metadata
        self.collection_metadata: Dict[str, Any] = {}
        
    def add_callback(self, callback: Callable[[List[CookieRecord]], None]) -> None:
        """Add callback to be called when cookies are collected.
        
        Args:
            callback: Function to call with cookie list
        """
        self._callbacks.append(callback)
    
    def _generate_cookie_key(self, cookie: Dict[str, Any]) -> str:
        """Generate unique key for cookie tracking.
        
        Args:
            cookie: Cookie dictionary from Playwright
            
        Returns:
            Unique cookie identifier
        """
        name = cookie.get('name', '')
        domain = cookie.get('domain', '').lstrip('.')
        path = cookie.get('path', '/')
        return f"{name}@{domain}{path}"
    
    def _classify_cookie_essential(self, cookie_name: str, cookie_domain: str, page_host: str) -> Optional[bool]:
        """Classify cookie as essential using configuration rules.
        
        Args:
            cookie_name: Name of the cookie
            cookie_domain: Domain of the cookie
            page_host: Host of the page
            
        Returns:
            True if essential, False if non-essential, None if unclear
        """
        if not self.config:
            return None
        
        # Check against essential patterns
        if self.config.is_essential_cookie(cookie_name):
            return True
            
        # Check for first-party authentication/session cookies
        cookie_domain_clean = cookie_domain.lstrip('.')
        is_first_party = (
            cookie_domain_clean == page_host or
            page_host.endswith(f'.{cookie_domain_clean}') or
            cookie_domain_clean.endswith(f'.{page_host}')
        )
        
        if is_first_party:
            # First-party cookies with common essential patterns
            essential_indicators = [
                'session', 'csrf', 'auth', 'login', 'security', 'consent', 'preferences'
            ]
            if any(indicator in cookie_name.lower() for indicator in essential_indicators):
                return True
        
        # Check against analytics patterns (non-essential)
        if self.config.is_analytics_cookie(cookie_name):
            return False
            
        # Check against analytics domains (non-essential)
        if self.config.is_analytics_domain(cookie_domain):
            return False
            
        # If unclear, return None (will be determined by other means)
        return None
    
    def _enhance_cookie_record(self, base_cookie: BaseCookieRecord, page_host: str, collection_time: datetime) -> CookieRecord:
        """Enhance base cookie record with privacy-specific metadata.
        
        Args:
            base_cookie: Base cookie record from existing collector
            page_host: Host of the page
            collection_time: When cookie was collected
            
        Returns:
            Enhanced cookie record with privacy metadata
        """
        # Generate metadata
        metadata = {
            'collection_time': collection_time.isoformat(),
            'collector': 'enhanced'
        }
        
        # Add configuration-based classification
        if self.config:
            metadata.update({
                'is_analytics': self.config.is_analytics_cookie(base_cookie.name) or 
                              self.config.is_analytics_domain(base_cookie.domain),
                'is_tracking': self.config.is_analytics_cookie(base_cookie.name),
                'analytics_domain': self.config.is_analytics_domain(base_cookie.domain)
            })
        
        # Determine essential status
        essential = self._classify_cookie_essential(base_cookie.name, base_cookie.domain, page_host)
        
        # Create enhanced record
        enhanced = CookieRecord(
            name=base_cookie.name,
            value=base_cookie.value,
            domain=base_cookie.domain,
            path=base_cookie.path,
            expires=base_cookie.expires,
            max_age=base_cookie.max_age,
            secure=base_cookie.secure,
            http_only=base_cookie.http_only,
            same_site=base_cookie.same_site,
            size=base_cookie.size,
            is_first_party=base_cookie.is_first_party,
            is_session=base_cookie.is_session,
            value_redacted=base_cookie.value_redacted,
            essential=essential,
            metadata=metadata,
            set_time=collection_time,
            modified_time=collection_time
        )
        
        return enhanced
    
    async def collect_initial_cookies(self, page_url: str) -> List[CookieRecord]:
        """Collect baseline cookies before any interactions.
        
        Args:
            page_url: URL of the page
            
        Returns:
            List of initial cookie records
        """
        try:
            page_host = urlparse(page_url).netloc
            collection_time = datetime.utcnow()
            
            # Get cookies from context
            playwright_cookies = await self.context.cookies()
            
            # Convert to enhanced records
            initial_cookies = []
            for pw_cookie in playwright_cookies:
                try:
                    # Create base cookie record
                    base_cookie = BaseCookieRecord.from_playwright_cookie(
                        pw_cookie, page_host, self.redact_values
                    )
                    
                    # Enhance with privacy metadata
                    enhanced_cookie = self._enhance_cookie_record(base_cookie, page_host, collection_time)
                    
                    # Track timeline
                    cookie_key = self._generate_cookie_key(pw_cookie)
                    if cookie_key not in self.cookie_timeline:
                        self.cookie_timeline[cookie_key] = []
                    self.cookie_timeline[cookie_key].append(collection_time)
                    
                    initial_cookies.append(enhanced_cookie)
                    
                except Exception as e:
                    logger.warning(f"Failed to process initial cookie {pw_cookie.get('name', 'unknown')}: {e}")
            
            self.initial_cookies = initial_cookies
            logger.info(f"Collected {len(initial_cookies)} initial cookies for {page_url}")
            
            return initial_cookies.copy()
            
        except Exception as e:
            logger.error(f"Failed to collect initial cookies: {e}")
            return []
    
    async def collect_cookies(self, page_url: str, scenario_id: Optional[str] = None) -> List[CookieRecord]:
        """Collect cookies with enhanced privacy tracking.
        
        Args:
            page_url: URL of the page
            scenario_id: Optional scenario identifier for tracking
            
        Returns:
            List of enhanced cookie records
        """
        try:
            page_host = urlparse(page_url).netloc
            collection_time = datetime.utcnow()
            
            # Get cookies from context
            playwright_cookies = await self.context.cookies()
            
            self.cookies = []
            
            for pw_cookie in playwright_cookies:
                try:
                    # Create base cookie record
                    base_cookie = BaseCookieRecord.from_playwright_cookie(
                        pw_cookie, page_host, self.redact_values
                    )
                    
                    # Enhance with privacy metadata
                    enhanced_cookie = self._enhance_cookie_record(base_cookie, page_host, collection_time)
                    
                    # Add scenario context if provided
                    if scenario_id:
                        enhanced_cookie.metadata['scenario_id'] = scenario_id
                    
                    # Update timeline tracking
                    cookie_key = self._generate_cookie_key(pw_cookie)
                    if cookie_key not in self.cookie_timeline:
                        self.cookie_timeline[cookie_key] = []
                    else:
                        # This is a modification, update modified_time
                        enhanced_cookie.modified_time = collection_time
                    
                    self.cookie_timeline[cookie_key].append(collection_time)
                    
                    self.cookies.append(enhanced_cookie)
                    
                except Exception as e:
                    logger.warning(f"Failed to process cookie {pw_cookie.get('name', 'unknown')}: {e}")
            
            # Update collection metadata
            self.collection_metadata.update({
                'last_collection_time': collection_time.isoformat(),
                'last_scenario_id': scenario_id,
                'cookies_collected': len(self.cookies)
            })
            
            # Call callbacks
            for callback in self._callbacks:
                try:
                    callback(self.cookies)
                except Exception as e:
                    logger.error(f"Error in cookie callback: {e}")
            
            logger.info(f"Collected {len(self.cookies)} cookies for {page_url} (scenario: {scenario_id})")
            return self.cookies.copy()
            
        except Exception as e:
            logger.error(f"Failed to collect enhanced cookies: {e}")
            return []
    
    async def detect_cookie_changes(self, page_url: str) -> Dict[str, List[CookieRecord]]:
        """Detect changes in cookies since initial collection.
        
        Args:
            page_url: URL of the page
            
        Returns:
            Dictionary with added, removed, and modified cookies
        """
        if not self.initial_cookies:
            await self.collect_initial_cookies(page_url)
            
        current_cookies = await self.collect_cookies(page_url)
        
        # Create lookup maps
        initial_map = {
            self._generate_cookie_key({
                'name': c.name, 
                'domain': c.domain, 
                'path': c.path
            }): c for c in self.initial_cookies
        }
        
        current_map = {
            self._generate_cookie_key({
                'name': c.name,
                'domain': c.domain,
                'path': c.path
            }): c for c in current_cookies
        }
        
        # Find differences
        added = []
        removed = []
        modified = []
        
        # Find added cookies
        for key, cookie in current_map.items():
            if key not in initial_map:
                added.append(cookie)
        
        # Find removed cookies  
        for key, cookie in initial_map.items():
            if key not in current_map:
                removed.append(cookie)
        
        # Find modified cookies
        for key in initial_map:
            if key in current_map:
                initial_cookie = initial_map[key]
                current_cookie = current_map[key]
                
                # Check for attribute changes
                if (initial_cookie.value != current_cookie.value or
                    initial_cookie.expires != current_cookie.expires or
                    initial_cookie.secure != current_cookie.secure or
                    initial_cookie.http_only != current_cookie.http_only or
                    initial_cookie.same_site != current_cookie.same_site):
                    modified.append(current_cookie)
        
        return {
            'added': added,
            'removed': removed,
            'modified': modified
        }
    
    def get_cookies_by_scenario(self, scenario_id: str) -> List[CookieRecord]:
        """Get cookies collected for a specific scenario.
        
        Args:
            scenario_id: Scenario identifier
            
        Returns:
            List of cookies from the specified scenario
        """
        return [
            cookie for cookie in self.cookies
            if cookie.metadata and cookie.metadata.get('scenario_id') == scenario_id
        ]
    
    def get_essential_cookies(self) -> List[CookieRecord]:
        """Get cookies classified as essential.
        
        Returns:
            List of essential cookies
        """
        return [cookie for cookie in self.cookies if cookie.essential is True]
    
    def get_non_essential_cookies(self) -> List[CookieRecord]:
        """Get cookies classified as non-essential.
        
        Returns:
            List of non-essential cookies
        """
        return [cookie for cookie in self.cookies if cookie.essential is False]
    
    def get_analytics_cookies(self) -> List[CookieRecord]:
        """Get cookies from analytics domains or with analytics patterns.
        
        Returns:
            List of analytics cookies
        """
        return [
            cookie for cookie in self.cookies
            if cookie.metadata and (
                cookie.metadata.get('is_analytics', False) or
                cookie.metadata.get('is_tracking', False)
            )
        ]
    
    def get_cookie_timeline(self, cookie_name: str, domain: str, path: str = "/") -> List[datetime]:
        """Get collection timeline for a specific cookie.
        
        Args:
            cookie_name: Cookie name
            domain: Cookie domain
            path: Cookie path
            
        Returns:
            List of timestamps when cookie was observed
        """
        cookie_key = f"{cookie_name}@{domain.lstrip('.')}{path}"
        return self.cookie_timeline.get(cookie_key, [])
    
    def get_cookie_stats(self) -> Dict[str, Any]:
        """Get comprehensive cookie statistics with privacy context.
        
        Returns:
            Dictionary with enhanced cookie statistics
        """
        if not self.cookies:
            return {}
        
        # Basic counts
        total_cookies = len(self.cookies)
        first_party = len([c for c in self.cookies if c.is_first_party])
        third_party = total_cookies - first_party
        
        # Essential classification
        essential = len(self.get_essential_cookies())
        non_essential = len(self.get_non_essential_cookies())
        unknown_essential = total_cookies - essential - non_essential
        
        # Analytics tracking
        analytics = len(self.get_analytics_cookies())
        
        # Security attributes
        secure_cookies = len([c for c in self.cookies if c.secure])
        http_only_cookies = len([c for c in self.cookies if c.http_only])
        same_site_cookies = len([c for c in self.cookies if c.same_site])
        
        # Timeline analysis
        unique_cookie_keys = len(self.cookie_timeline)
        total_observations = sum(len(timestamps) for timestamps in self.cookie_timeline.values())
        
        # Size analysis
        total_size = sum(cookie.size for cookie in self.cookies)
        
        return {
            'total_cookies': total_cookies,
            'first_party_cookies': first_party,
            'third_party_cookies': third_party,
            'essential_cookies': essential,
            'non_essential_cookies': non_essential,
            'unknown_essential_status': unknown_essential,
            'analytics_cookies': analytics,
            'security_attributes': {
                'secure_cookies': secure_cookies,
                'http_only_cookies': http_only_cookies,
                'same_site_cookies': same_site_cookies,
                'secure_percentage': (secure_cookies / total_cookies * 100) if total_cookies > 0 else 0,
            },
            'timeline_analysis': {
                'unique_cookies_observed': unique_cookie_keys,
                'total_observations': total_observations,
                'average_observations_per_cookie': (total_observations / unique_cookie_keys) if unique_cookie_keys > 0 else 0,
            },
            'size_analysis': {
                'total_size_bytes': total_size,
                'average_cookie_size': total_size / total_cookies if total_cookies > 0 else 0,
            },
            'collection_metadata': self.collection_metadata
        }
    
    def export_privacy_summary(self) -> Dict[str, Any]:
        """Export privacy-focused summary of cookie collection.
        
        Returns:
            Privacy analysis summary
        """
        stats = self.get_cookie_stats()
        
        privacy_score = 0
        max_score = 100
        
        # Calculate privacy score based on various factors
        if stats['total_cookies'] > 0:
            # Fewer third-party cookies is better
            third_party_ratio = stats['third_party_cookies'] / stats['total_cookies']
            privacy_score += (1 - third_party_ratio) * 30  # Up to 30 points
            
            # Fewer analytics cookies is better  
            analytics_ratio = stats['analytics_cookies'] / stats['total_cookies']
            privacy_score += (1 - analytics_ratio) * 25  # Up to 25 points
            
            # Better security attributes
            security_attrs = stats['security_attributes']
            privacy_score += security_attrs['secure_percentage'] * 0.2  # Up to 20 points
            
            # Essential vs non-essential ratio
            if stats['non_essential_cookies'] > 0:
                essential_ratio = stats['essential_cookies'] / (stats['essential_cookies'] + stats['non_essential_cookies'])
                privacy_score += essential_ratio * 25  # Up to 25 points
            else:
                privacy_score += 25  # All essential
        
        return {
            'privacy_score': min(privacy_score, max_score),
            'cookie_breakdown': {
                'total': stats['total_cookies'],
                'first_party': stats['first_party_cookies'],
                'third_party': stats['third_party_cookies'],
                'essential': stats['essential_cookies'],
                'non_essential': stats['non_essential_cookies'],
                'analytics': stats['analytics_cookies'],
            },
            'privacy_indicators': {
                'third_party_ratio': (stats['third_party_cookies'] / stats['total_cookies']) if stats['total_cookies'] > 0 else 0,
                'analytics_ratio': (stats['analytics_cookies'] / stats['total_cookies']) if stats['total_cookies'] > 0 else 0,
                'secure_ratio': stats['security_attributes']['secure_percentage'] / 100,
                'essential_ratio': (stats['essential_cookies'] / stats['total_cookies']) if stats['total_cookies'] > 0 else 0,
            },
            'recommendations': self._generate_recommendations(stats)
        }
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate privacy recommendations based on cookie analysis.
        
        Args:
            stats: Cookie statistics
            
        Returns:
            List of privacy improvement recommendations
        """
        recommendations = []
        
        if stats['total_cookies'] == 0:
            return recommendations
        
        # High third-party cookie usage
        third_party_ratio = stats['third_party_cookies'] / stats['total_cookies']
        if third_party_ratio > 0.5:
            recommendations.append("Consider reducing third-party cookie usage for better privacy")
        
        # High analytics cookie usage
        analytics_ratio = stats['analytics_cookies'] / stats['total_cookies']
        if analytics_ratio > 0.3:
            recommendations.append("Consider implementing consent management for analytics cookies")
        
        # Security attribute issues
        security_attrs = stats['security_attributes']
        if security_attrs['secure_percentage'] < 80:
            recommendations.append("Add Secure flag to cookies on HTTPS sites")
        
        if security_attrs['http_only_cookies'] < stats['essential_cookies']:
            recommendations.append("Add HttpOnly flag to session cookies for security")
        
        # Non-essential cookies without consent
        if stats['non_essential_cookies'] > stats['essential_cookies']:
            recommendations.append("Implement consent management for non-essential cookies")
        
        return recommendations
    
    def clear(self) -> None:
        """Clear all collected data and reset state."""
        self.cookies.clear()
        self.cookie_timeline.clear()
        self.initial_cookies = None
        self.collection_metadata.clear()
        logger.debug("Enhanced cookie collector cleared")
    
    def __repr__(self) -> str:
        """String representation of enhanced cookie collector."""
        stats = self.get_cookie_stats()
        if not stats:
            return "EnhancedCookieCollector(no data)"
        
        return (
            f"EnhancedCookieCollector("
            f"total={stats['total_cookies']}, "
            f"first_party={stats['first_party_cookies']}, "
            f"third_party={stats['third_party_cookies']}, "
            f"essential={stats['essential_cookies']}, "
            f"analytics={stats['analytics_cookies']})"
        )