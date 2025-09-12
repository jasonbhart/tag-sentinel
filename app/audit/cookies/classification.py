"""Cookie classification system for privacy analysis.

This module provides sophisticated cookie classification based on domain analysis,
eTLD+1 comparison, and privacy policy requirements. Supports first-party vs third-party
classification, essential cookie detection, and policy compliance checking.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urlparse
from enum import Enum

from .models import CookieRecord, CookiePolicyIssue, PolicySeverity
from .config import PrivacyConfiguration

logger = logging.getLogger(__name__)


class CookieCategory(str, Enum):
    """Cookie categories for classification."""
    ESSENTIAL = "essential"
    FUNCTIONAL = "functional"
    ANALYTICS = "analytics"
    MARKETING = "marketing"
    SOCIAL = "social"
    PREFERENCE = "preference"
    UNKNOWN = "unknown"


class DomainScope(str, Enum):
    """Domain relationship types."""
    SAME_SITE = "same_site"          # Exact same domain
    SUBDOMAIN = "subdomain"          # Subdomain of page domain
    SUPERDOMAIN = "superdomain"      # Page is subdomain of cookie domain
    RELATED = "related"              # Same eTLD+1 but different subdomains
    THIRD_PARTY = "third_party"      # Different eTLD+1


class CookieClassifier:
    """Advanced cookie classification system with eTLD+1 analysis.
    
    Provides sophisticated domain analysis, cookie categorization,
    and policy compliance checking for privacy analysis.
    """
    
    def __init__(self, config: Optional[PrivacyConfiguration] = None):
        """Initialize cookie classifier.
        
        Args:
            config: Privacy configuration with classification rules
        """
        self.config = config
        
        # Common public suffixes for eTLD+1 analysis
        # This is a simplified list - in production would use Mozilla's PSL
        self.public_suffixes = {
            'com', 'org', 'net', 'edu', 'gov', 'mil', 'int',
            'co.uk', 'org.uk', 'ac.uk', 'gov.uk', 'com.au', 'org.au',
            'co.jp', 'ne.jp', 'or.jp', 'ac.jp', 'ad.jp', 'co.in',
            'com.br', 'org.br', 'gov.br', 'mil.br', 'net.br',
            'com.cn', 'org.cn', 'net.cn', 'ac.cn', 'ah.cn',
            'co.za', 'org.za', 'net.za', 'ac.za', 'gov.za',
            'github.io', 'herokuapp.com', 'appspot.com', 'amazonaws.com'
        }
        
        # Well-known CDN and service domains that are often first-party
        self.cdn_domains = {
            'cloudfront.net', 'cloudflare.com', 'fastly.com', 
            'jsdelivr.net', 'unpkg.com', 'cdnjs.cloudflare.com',
            'maxcdn.bootstrapcdn.com', 'ajax.googleapis.com',
            'fonts.googleapis.com', 'fonts.gstatic.com'
        }
        
        # SSO and authentication domains
        self.sso_domains = {
            'accounts.google.com', 'login.microsoftonline.com',
            'auth0.com', 'okta.com', 'salesforce.com',
            'facebook.com', 'twitter.com', 'linkedin.com'  # OAuth providers
        }
        
    def _extract_etld_plus_one(self, domain: str) -> str:
        """Extract eTLD+1 (effective top-level domain + 1) from domain.
        
        Args:
            domain: Domain to analyze
            
        Returns:
            eTLD+1 string
        """
        if not domain:
            return ""
        
        # Remove leading dot and convert to lowercase
        clean_domain = domain.lstrip('.').lower()
        
        # Handle IP addresses
        if self._is_ip_address(clean_domain):
            return clean_domain
        
        # Split into parts
        parts = clean_domain.split('.')
        if len(parts) <= 1:
            return clean_domain
        
        # Check for known multi-part suffixes first (longest match)
        for i in range(len(parts) - 1):
            suffix_candidate = '.'.join(parts[i:])
            if suffix_candidate in self.public_suffixes:
                # Found a public suffix, eTLD+1 includes one more label
                if i > 0:
                    return '.'.join(parts[i-1:])
                else:
                    # The entire domain is a public suffix
                    return clean_domain
        
        # Check for single-part suffixes
        if parts[-1] in self.public_suffixes:
            if len(parts) >= 2:
                return '.'.join(parts[-2:])
            else:
                return clean_domain
        
        # Default: assume standard TLD, return domain.tld
        if len(parts) >= 2:
            return '.'.join(parts[-2:])
        
        return clean_domain
    
    def _is_ip_address(self, domain: str) -> bool:
        """Check if domain is an IP address.
        
        Args:
            domain: Domain to check
            
        Returns:
            True if domain is an IP address
        """
        # Simple IPv4 check
        parts = domain.split('.')
        if len(parts) == 4:
            try:
                return all(0 <= int(part) <= 255 for part in parts)
            except ValueError:
                pass
        
        # IPv6 check (simplified)
        if ':' in domain:
            return True
        
        return False
    
    def _determine_domain_relationship(self, cookie_domain: str, page_domain: str) -> DomainScope:
        """Determine relationship between cookie domain and page domain.
        
        Args:
            cookie_domain: Domain where cookie is set
            page_domain: Domain of the page
            
        Returns:
            Domain relationship classification
        """
        clean_cookie_domain = cookie_domain.lstrip('.').lower()
        clean_page_domain = page_domain.lower()
        
        # Exact match
        if clean_cookie_domain == clean_page_domain:
            return DomainScope.SAME_SITE
        
        # Check subdomain relationships
        if clean_page_domain.endswith(f'.{clean_cookie_domain}'):
            return DomainScope.SUPERDOMAIN
        
        if clean_cookie_domain.endswith(f'.{clean_page_domain}'):
            return DomainScope.SUBDOMAIN
        
        # Check eTLD+1 relationship
        cookie_etld1 = self._extract_etld_plus_one(clean_cookie_domain)
        page_etld1 = self._extract_etld_plus_one(clean_page_domain)
        
        if cookie_etld1 == page_etld1 and cookie_etld1:
            return DomainScope.RELATED
        
        return DomainScope.THIRD_PARTY
    
    def _is_cdn_or_service_domain(self, domain: str) -> bool:
        """Check if domain is a known CDN or service domain.
        
        Args:
            domain: Domain to check
            
        Returns:
            True if domain is a CDN or service
        """
        clean_domain = domain.lstrip('.').lower()
        
        # Direct match
        if clean_domain in self.cdn_domains:
            return True
        
        # Check if it's a subdomain of a CDN
        for cdn_domain in self.cdn_domains:
            if clean_domain.endswith(f'.{cdn_domain}'):
                return True
        
        return False
    
    def _is_sso_domain(self, domain: str) -> bool:
        """Check if domain is a known SSO/authentication domain.
        
        Args:
            domain: Domain to check
            
        Returns:
            True if domain is for SSO/authentication
        """
        clean_domain = domain.lstrip('.').lower()
        
        for sso_domain in self.sso_domains:
            if clean_domain == sso_domain or clean_domain.endswith(f'.{sso_domain}'):
                return True
        
        return False
    
    def classify_cookie_party(self, cookie: CookieRecord, page_url: str) -> Tuple[bool, DomainScope, Dict[str, Any]]:
        """Classify cookie as first-party or third-party with detailed analysis.
        
        Args:
            cookie: Cookie to classify
            page_url: URL of the page where cookie was found
            
        Returns:
            Tuple of (is_first_party, domain_scope, analysis_details)
        """
        page_domain = urlparse(page_url).netloc.lower()
        cookie_domain = cookie.domain
        
        # Determine domain relationship
        domain_scope = self._determine_domain_relationship(cookie_domain, page_domain)
        
        # Base classification
        is_first_party = domain_scope in [DomainScope.SAME_SITE, DomainScope.SUBDOMAIN, DomainScope.RELATED]
        
        # Special cases that affect classification
        analysis_details = {
            'page_domain': page_domain,
            'cookie_domain': cookie_domain,
            'domain_scope': domain_scope.value,
            'page_etld1': self._extract_etld_plus_one(page_domain),
            'cookie_etld1': self._extract_etld_plus_one(cookie_domain),
            'special_cases': []
        }
        
        # CDN/Service domain adjustment
        if domain_scope == DomainScope.THIRD_PARTY and self._is_cdn_or_service_domain(cookie_domain):
            # CDN cookies can be considered first-party for functionality
            analysis_details['special_cases'].append('cdn_service_domain')
            # Keep as third-party for privacy analysis, but note the CDN status
        
        # SSO domain consideration  
        if domain_scope == DomainScope.THIRD_PARTY and self._is_sso_domain(cookie_domain):
            analysis_details['special_cases'].append('sso_authentication_domain')
            # SSO cookies might be necessary for authentication flows
        
        # Superdomain case (cookie set on parent domain)
        if domain_scope == DomainScope.SUPERDOMAIN:
            # This is typically first-party (parent domain setting cookie for subdomain)
            is_first_party = True
            analysis_details['special_cases'].append('parent_domain_cookie')
        
        return is_first_party, domain_scope, analysis_details
    
    def classify_cookie_category(self, cookie: CookieRecord, page_url: str) -> Tuple[CookieCategory, float]:
        """Classify cookie into functional category with confidence score.
        
        Args:
            cookie: Cookie to classify
            page_url: URL of the page
            
        Returns:
            Tuple of (category, confidence_score)
        """
        cookie_name = cookie.name.lower()
        cookie_domain = cookie.domain.lower()
        
        # Essential/Necessary cookies (high confidence patterns)
        essential_patterns = [
            (r'^session.*', CookieCategory.ESSENTIAL, 0.95),
            (r'^jsessionid$', CookieCategory.ESSENTIAL, 0.98),
            (r'^phpsessid$', CookieCategory.ESSENTIAL, 0.98),
            (r'^asp\.net_sessionid$', CookieCategory.ESSENTIAL, 0.98),
            (r'^csrf.*', CookieCategory.ESSENTIAL, 0.95),
            (r'^authenticity_token$', CookieCategory.ESSENTIAL, 0.95),
            (r'^auth.*', CookieCategory.ESSENTIAL, 0.85),
            (r'^login.*', CookieCategory.ESSENTIAL, 0.85),
            (r'^security.*', CookieCategory.ESSENTIAL, 0.80),
        ]
        
        # Analytics cookies
        analytics_patterns = [
            (r'^_ga.*', CookieCategory.ANALYTICS, 0.98),
            (r'^_gid$', CookieCategory.ANALYTICS, 0.98),
            (r'^_gat.*', CookieCategory.ANALYTICS, 0.95),
            (r'^__utm.*', CookieCategory.ANALYTICS, 0.95),
            (r'^_dc_gtm_.*', CookieCategory.ANALYTICS, 0.90),
            (r'^_hjid$', CookieCategory.ANALYTICS, 0.95),
            (r'^_hjSessionUser_.*', CookieCategory.ANALYTICS, 0.95),
            (r'^mp_.*', CookieCategory.ANALYTICS, 0.90),
        ]
        
        # Marketing/Advertising cookies
        marketing_patterns = [
            (r'^_fbp$', CookieCategory.MARKETING, 0.95),
            (r'^_fbc$', CookieCategory.MARKETING, 0.95),
            (r'^fr$', CookieCategory.MARKETING, 0.90),  # Facebook
            (r'^ide$', CookieCategory.MARKETING, 0.85),  # Google DoubleClick
            (r'^test_cookie$', CookieCategory.MARKETING, 0.80),
            (r'^ads.*', CookieCategory.MARKETING, 0.80),
            (r'^doubleclick.*', CookieCategory.MARKETING, 0.85),
        ]
        
        # Social media cookies
        social_patterns = [
            (r'^_twitter_sess$', CookieCategory.SOCIAL, 0.90),
            (r'^guest_id$', CookieCategory.SOCIAL, 0.85),
            (r'^li_.*', CookieCategory.SOCIAL, 0.85),  # LinkedIn
            (r'^bcookie$', CookieCategory.SOCIAL, 0.85),  # LinkedIn
        ]
        
        # Preference/Functional cookies
        preference_patterns = [
            (r'^pref.*', CookieCategory.PREFERENCE, 0.85),
            (r'^settings.*', CookieCategory.PREFERENCE, 0.85),
            (r'^theme.*', CookieCategory.PREFERENCE, 0.80),
            (r'^lang.*', CookieCategory.PREFERENCE, 0.80),
            (r'^locale.*', CookieCategory.PREFERENCE, 0.80),
            (r'^timezone.*', CookieCategory.PREFERENCE, 0.80),
            (r'^consent.*', CookieCategory.PREFERENCE, 0.90),
            (r'^cookie.*', CookieCategory.PREFERENCE, 0.75),
        ]
        
        # Check patterns in order of specificity
        all_patterns = (
            essential_patterns + analytics_patterns + 
            marketing_patterns + social_patterns + preference_patterns
        )
        
        for pattern, category, confidence in all_patterns:
            if re.match(pattern, cookie_name):
                return category, confidence
        
        # Domain-based classification
        analytics_domains = [
            'google-analytics.com', 'googletagmanager.com', 'hotjar.com', 
            'mixpanel.com', 'segment.com', 'fullstory.com', 'amplitude.com'
        ]
        
        marketing_domains = [
            'doubleclick.net', 'facebook.com', 'adsystem.amazon.com',
            'bing.com', 'twitter.com', 'linkedin.com', 'pinterest.com'
        ]
        
        for domain in analytics_domains:
            if domain in cookie_domain:
                return CookieCategory.ANALYTICS, 0.80
        
        for domain in marketing_domains:
            if domain in cookie_domain:
                return CookieCategory.MARKETING, 0.80
        
        # Default classification based on party status
        is_first_party, _, _ = self.classify_cookie_party(cookie, page_url)
        
        if is_first_party:
            # First-party cookies are more likely to be functional
            if cookie.is_session:
                return CookieCategory.ESSENTIAL, 0.60
            else:
                return CookieCategory.FUNCTIONAL, 0.50
        else:
            # Third-party cookies are more likely to be marketing/analytics
            return CookieCategory.MARKETING, 0.40
    
    def classify_cookies(self, cookies: List[CookieRecord], page_url: str) -> List[CookieRecord]:
        """Classify a list of cookies with comprehensive analysis.
        
        Args:
            cookies: List of cookies to classify
            page_url: URL of the page
            
        Returns:
            List of cookies with updated classification metadata
        """
        classified_cookies = []
        
        for cookie in cookies:
            # Classify party status
            is_first_party, domain_scope, party_analysis = self.classify_cookie_party(cookie, page_url)
            
            # Classify category
            category, confidence = self.classify_cookie_category(cookie, page_url)
            
            # Update cookie with classification
            updated_cookie = cookie.model_copy()
            updated_cookie.is_first_party = is_first_party
            
            # Update metadata
            if not updated_cookie.metadata:
                updated_cookie.metadata = {}
            
            updated_cookie.metadata.update({
                'classification': {
                    'category': category.value,
                    'category_confidence': confidence,
                    'domain_scope': domain_scope.value,
                    'party_analysis': party_analysis,
                    'is_essential_heuristic': category == CookieCategory.ESSENTIAL,
                    'is_analytics_heuristic': category == CookieCategory.ANALYTICS,
                    'is_marketing_heuristic': category == CookieCategory.MARKETING,
                }
            })
            
            # Set essential status based on category if not already set
            if updated_cookie.essential is None:
                updated_cookie.essential = category == CookieCategory.ESSENTIAL
            
            classified_cookies.append(updated_cookie)
        
        logger.info(f"Classified {len(classified_cookies)} cookies for {page_url}")
        
        return classified_cookies
    
    def check_policies(self, cookies: List[CookieRecord], page_url: str, environment: str = "production") -> List[CookiePolicyIssue]:
        """Check cookies against privacy policy requirements.
        
        Args:
            cookies: List of cookies to check
            page_url: URL of the page
            environment: Environment context (affects policy strictness)
            
        Returns:
            List of policy violations found
        """
        issues = []
        page_scheme = urlparse(page_url).scheme
        
        for cookie in cookies:
            # Check Secure flag on HTTPS sites
            if page_scheme == 'https' and not cookie.secure:
                severity = PolicySeverity.HIGH if environment == 'production' else PolicySeverity.MEDIUM
                
                issues.append(CookiePolicyIssue(
                    cookie_name=cookie.name,
                    cookie_domain=cookie.domain,
                    cookie_path=cookie.path,
                    attribute="secure",
                    expected="true",
                    observed="false",
                    severity=severity,
                    rule_id="secure_flag_https",
                    message=f"Cookie {cookie.name} lacks Secure flag on HTTPS site",
                    page_url=page_url,
                    remediation="Add Secure flag to cookies on HTTPS sites to prevent transmission over HTTP"
                ))
            
            # Check SameSite attribute
            if not cookie.same_site or cookie.same_site.lower() == 'none':
                severity = PolicySeverity.MEDIUM
                expected_same_site = "Lax"
                
                # Third-party cookies might legitimately use None, but should be flagged
                if not cookie.is_first_party and cookie.same_site and cookie.same_site.lower() == 'none':
                    severity = PolicySeverity.LOW
                    expected_same_site = "None (with Secure flag)"
                
                issues.append(CookiePolicyIssue(
                    cookie_name=cookie.name,
                    cookie_domain=cookie.domain,
                    cookie_path=cookie.path,
                    attribute="same_site",
                    expected=expected_same_site,
                    observed=cookie.same_site or "not_set",
                    severity=severity,
                    rule_id="same_site_required",
                    message=f"Cookie {cookie.name} has inadequate SameSite protection",
                    page_url=page_url,
                    remediation="Set SameSite=Lax for most cookies, or SameSite=Strict for sensitive cookies"
                ))
            
            # Check HttpOnly for session cookies
            if (cookie.is_session or 'session' in cookie.name.lower() or 
                (cookie.essential and 'auth' in cookie.name.lower())) and not cookie.http_only:
                
                issues.append(CookiePolicyIssue(
                    cookie_name=cookie.name,
                    cookie_domain=cookie.domain,
                    cookie_path=cookie.path,
                    attribute="http_only",
                    expected="true",
                    observed="false",
                    severity=PolicySeverity.HIGH,
                    rule_id="http_only_sessions",
                    message=f"Session/authentication cookie {cookie.name} lacks HttpOnly flag",
                    page_url=page_url,
                    remediation="Add HttpOnly flag to session cookies to prevent XSS access"
                ))
            
            # Check for analytics cookies that might need consent
            if (cookie.metadata and 
                cookie.metadata.get('classification', {}).get('category') == 'analytics' and
                not cookie.essential):
                
                issues.append(CookiePolicyIssue(
                    cookie_name=cookie.name,
                    cookie_domain=cookie.domain,
                    cookie_path=cookie.path,
                    attribute="consent",
                    expected="required",
                    observed="not_verified",
                    severity=PolicySeverity.MEDIUM,
                    rule_id="analytics_consent_required",
                    message=f"Analytics cookie {cookie.name} may require user consent",
                    page_url=page_url,
                    remediation="Implement consent management for analytics cookies per privacy regulations"
                ))
        
        logger.info(f"Found {len(issues)} policy issues for {len(cookies)} cookies")
        return issues
    
    def generate_classification_report(self, cookies: List[CookieRecord], page_url: str) -> Dict[str, Any]:
        """Generate comprehensive classification report.
        
        Args:
            cookies: Classified cookies
            page_url: URL of the page
            
        Returns:
            Detailed classification analysis report
        """
        if not cookies:
            return {'total_cookies': 0, 'summary': 'No cookies found'}
        
        # Count by party
        first_party = [c for c in cookies if c.is_first_party]
        third_party = [c for c in cookies if not c.is_first_party]
        
        # Count by category
        category_counts = {}
        for cookie in cookies:
            category = cookie.metadata.get('classification', {}).get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Count by domain scope
        scope_counts = {}
        for cookie in cookies:
            scope = cookie.metadata.get('classification', {}).get('domain_scope', 'unknown')
            scope_counts[scope] = scope_counts.get(scope, 0) + 1
        
        # Security analysis
        secure_cookies = [c for c in cookies if c.secure]
        http_only_cookies = [c for c in cookies if c.http_only]
        same_site_cookies = [c for c in cookies if c.same_site]
        
        # Essential vs non-essential
        essential_cookies = [c for c in cookies if c.essential]
        non_essential_cookies = [c for c in cookies if c.essential is False]
        
        return {
            'total_cookies': len(cookies),
            'party_breakdown': {
                'first_party': len(first_party),
                'third_party': len(third_party),
                'first_party_percentage': (len(first_party) / len(cookies)) * 100,
            },
            'category_breakdown': category_counts,
            'domain_scope_breakdown': scope_counts,
            'security_attributes': {
                'secure_count': len(secure_cookies),
                'http_only_count': len(http_only_cookies),
                'same_site_count': len(same_site_cookies),
                'secure_percentage': (len(secure_cookies) / len(cookies)) * 100,
            },
            'essential_classification': {
                'essential_count': len(essential_cookies),
                'non_essential_count': len(non_essential_cookies),
                'unknown_count': len(cookies) - len(essential_cookies) - len(non_essential_cookies),
                'essential_percentage': (len(essential_cookies) / len(cookies)) * 100 if essential_cookies else 0,
            },
            'page_context': {
                'page_url': page_url,
                'page_domain': urlparse(page_url).netloc,
                'page_etld1': self._extract_etld_plus_one(urlparse(page_url).netloc),
            }
        }