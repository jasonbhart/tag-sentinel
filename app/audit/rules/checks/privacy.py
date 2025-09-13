"""Privacy and cookie compliance checks for GDPR, CCPA, and security validation.

This module implements comprehensive privacy compliance checks to ensure websites
meet GDPR, CCPA, and other privacy regulation requirements, along with cookie
security best practices validation.
"""

import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from ...models.capture import RequestLog, CookieRecord
from ...detectors.base import TagEvent
from ..models import Severity
from .base import BaseCheck, CheckContext, CheckResult, register_check


class PrivacyRegulation:
    """Privacy regulation constants and helpers."""
    
    GDPR_ESSENTIAL_PURPOSES = {
        'strictly_necessary',
        'security',
        'authentication',
        'load_balancing',
        'fraud_prevention'
    }
    
    CCPA_SENSITIVE_CATEGORIES = {
        'personal_info',
        'financial_info',
        'health_info',
        'biometric_data',
        'location_data',
        'browsing_history'
    }
    
    COOKIE_SECURITY_ATTRIBUTES = {
        'secure': 'Cookie should have Secure flag when served over HTTPS',
        'http_only': 'Cookie should have HttpOnly flag to prevent XSS',
        'same_site': 'Cookie should have SameSite attribute to prevent CSRF'
    }


@register_check("gdpr_compliance")
class GDPRComplianceCheck(BaseCheck):
    """Validate GDPR compliance requirements for cookies and data collection."""
    
    def __init__(self, check_id: str, name: str, description: str):
        super().__init__(check_id, name, description)
    
    def get_supported_config_keys(self) -> List[str]:
        return [
            'essential_cookie_patterns',
            'consent_required_patterns',
            'consent_detection',
            'lawful_basis_mapping',
            'data_retention_limits',
            'cross_border_transfers',
            'cookie_categorization'
        ]
    
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute GDPR compliance validation."""
        config = context.check_config
        
        violations = []
        
        # Check cookie categorization and consent requirements
        cookie_violations = self._check_cookie_consent_requirements(context, config)
        violations.extend(cookie_violations)
        
        # Check consent mechanism presence
        consent_violations = self._check_consent_mechanism(context, config)
        violations.extend(consent_violations)
        
        # Check data retention compliance
        retention_violations = self._check_data_retention(context, config)
        violations.extend(retention_violations)
        
        # Check cross-border transfer compliance
        transfer_violations = self._check_cross_border_transfers(context, config)
        violations.extend(transfer_violations)
        
        passed = len(violations) == 0
        message = f"Found {len(violations)} GDPR compliance violations"
        
        return self._create_result(
            context=context,
            passed=passed,
            message=message,
            found_count=len(violations),
            expected_count=0,
            evidence=violations[:20]
        )
    
    def _check_cookie_consent_requirements(
        self, 
        context: CheckContext, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check that cookies requiring consent have proper mechanisms."""
        violations = []
        
        essential_patterns = config.get('essential_cookie_patterns', [])
        consent_required_patterns = config.get('consent_required_patterns', [])
        cookie_categorization = config.get('cookie_categorization', {})
        
        # Get all cookies
        cookies = context.query.query().cookies()
        
        for cookie in cookies:
            # Determine if cookie is essential
            is_essential = self._is_essential_cookie(cookie, essential_patterns)
            
            # Check if cookie requires consent
            requires_consent = (
                not is_essential or 
                self._matches_patterns(cookie.name, consent_required_patterns)
            )
            
            if requires_consent:
                # Verify consent mechanism exists
                if not self._has_consent_for_cookie(cookie, context, config):
                    violations.append({
                        'type': 'missing_cookie_consent',
                        'cookie_name': cookie.name,
                        'cookie_domain': cookie.domain,
                        'is_essential': is_essential,
                        'category': cookie_categorization.get(cookie.name, 'uncategorized'),
                        'lawful_basis_required': True
                    })
            
            # Check for proper categorization
            if cookie.name not in cookie_categorization:
                violations.append({
                    'type': 'uncategorized_cookie',
                    'cookie_name': cookie.name,
                    'cookie_domain': cookie.domain,
                    'requires_categorization': True
                })
        
        return violations
    
    def _check_consent_mechanism(
        self, 
        context: CheckContext, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for presence and validity of consent mechanisms."""
        violations = []
        
        consent_detection = config.get('consent_detection', {})
        
        # Look for consent management platform (CMP) requests
        cmp_requests = context.query.query().where_regex(
            'url', r'(consent|cmp|cookie.*banner|gdpr)'
        ).requests()
        
        if not cmp_requests:
            violations.append({
                'type': 'missing_consent_mechanism',
                'description': 'No consent management platform detected',
                'requirement': 'GDPR requires clear consent for non-essential cookies'
            })
        
        # Check for consent-related cookies
        consent_cookies = context.query.query().where_regex(
            'name', r'(consent|gdpr|cookieconsent|cmp)'
        ).cookies()
        
        if not consent_cookies:
            violations.append({
                'type': 'missing_consent_storage',
                'description': 'No consent preference storage detected',
                'requirement': 'Consent choices must be stored and respected'
            })
        
        # Check for consent withdrawal mechanism
        if consent_detection.get('withdrawal_required', True):
            withdrawal_detected = self._detect_consent_withdrawal(context)
            if not withdrawal_detected:
                violations.append({
                    'type': 'missing_consent_withdrawal',
                    'description': 'No consent withdrawal mechanism detected',
                    'requirement': 'GDPR requires easy consent withdrawal'
                })
        
        return violations
    
    def _check_data_retention(
        self, 
        context: CheckContext, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check data retention compliance."""
        violations = []
        
        retention_limits = config.get('data_retention_limits', {})
        
        # Check cookie expiration times
        cookies = context.query.query().cookies()
        
        for cookie in cookies:
            if not cookie.is_session and cookie.expires:
                cookie_category = self._categorize_cookie(cookie, config)
                retention_limit_days = retention_limits.get(cookie_category)
                
                if retention_limit_days:
                    cookie_duration_days = (cookie.expires - datetime.utcnow()).days
                    
                    if cookie_duration_days > retention_limit_days:
                        violations.append({
                            'type': 'excessive_data_retention',
                            'cookie_name': cookie.name,
                            'category': cookie_category,
                            'retention_days': cookie_duration_days,
                            'limit_days': retention_limit_days,
                            'excess_days': cookie_duration_days - retention_limit_days
                        })
        
        return violations
    
    def _check_cross_border_transfers(
        self, 
        context: CheckContext, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check cross-border data transfer compliance."""
        violations = []
        
        transfer_config = config.get('cross_border_transfers', {})
        allowed_countries = set(transfer_config.get('allowed_countries', []))
        adequacy_countries = set(transfer_config.get('adequacy_countries', [
            'US', 'CA', 'JP', 'KR', 'NZ', 'CH', 'IL', 'AD', 'AR', 'UY'  # Sample adequacy list
        ]))
        
        # Analyze third-party requests for potential data transfers
        requests = context.query.query().requests()
        
        for request in requests:
            domain = urlparse(request.url).netloc
            
            # Simple country detection based on TLD (basic implementation)
            country_code = self._extract_country_from_domain(domain)
            
            if country_code and country_code not in allowed_countries:
                if country_code not in adequacy_countries:
                    violations.append({
                        'type': 'unauthorized_cross_border_transfer',
                        'domain': domain,
                        'country_code': country_code,
                        'requires_adequacy_decision': True,
                        'requires_safeguards': True
                    })
        
        return violations
    
    def _is_essential_cookie(self, cookie: CookieRecord, essential_patterns: List[str]) -> bool:
        """Determine if cookie is strictly necessary/essential."""
        # Check against essential patterns
        if self._matches_patterns(cookie.name, essential_patterns):
            return True
        
        # Common essential cookie patterns
        essential_cookie_names = {
            'PHPSESSID', 'JSESSIONID', 'ASP.NET_SessionId', 'CFID', 'CFTOKEN',
            '__RequestVerificationToken', 'csrftoken', '_token',
            'wordpress_logged_in', 'wp-settings', '_wpnonce',
            '__Secure-', '__Host-'  # Secure cookie prefixes
        }
        
        return any(pattern in cookie.name for pattern in essential_cookie_names)
    
    def _matches_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the regex patterns."""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _has_consent_for_cookie(
        self, 
        cookie: CookieRecord, 
        context: CheckContext, 
        config: Dict[str, Any]
    ) -> bool:
        """Check if proper consent exists for the cookie."""
        # This is a simplified check - in practice, would need more sophisticated analysis
        consent_cookies = context.query.query().where_regex(
            'name', r'(consent|gdpr|cookieconsent)'
        ).cookies()
        
        return len(consent_cookies) > 0
    
    def _detect_consent_withdrawal(self, context: CheckContext) -> bool:
        """Detect presence of consent withdrawal mechanism."""
        # Look for privacy policy or cookie settings requests
        privacy_requests = context.query.query().where_regex(
            'url', r'(privacy|cookie.*settings|preferences|opt.*out)'
        ).requests()
        
        return len(privacy_requests) > 0
    
    def _categorize_cookie(self, cookie: CookieRecord, config: Dict[str, Any]) -> str:
        """Categorize cookie based on configuration."""
        categorization = config.get('cookie_categorization', {})
        return categorization.get(cookie.name, 'uncategorized')
    
    def _extract_country_from_domain(self, domain: str) -> Optional[str]:
        """Extract country code from domain (basic TLD-based detection)."""
        # Very basic implementation - real-world would use IP geolocation
        tld_to_country = {
            '.uk': 'GB', '.de': 'DE', '.fr': 'FR', '.it': 'IT', '.es': 'ES',
            '.nl': 'NL', '.be': 'BE', '.pl': 'PL', '.se': 'SE', '.dk': 'DK',
            '.com': 'US', '.org': 'US', '.net': 'US',  # Assuming US for generic
            '.cn': 'CN', '.ru': 'RU', '.jp': 'JP', '.kr': 'KR',
            '.au': 'AU', '.ca': 'CA', '.br': 'BR', '.mx': 'MX'
        }
        
        for tld, country in tld_to_country.items():
            if domain.endswith(tld):
                return country
        
        return None


@register_check("ccpa_compliance")
class CCPAComplianceCheck(BaseCheck):
    """Validate CCPA compliance requirements for consumer privacy rights."""
    
    def __init__(self, check_id: str, name: str, description: str):
        super().__init__(check_id, name, description)
    
    def get_supported_config_keys(self) -> List[str]:
        return [
            'personal_info_collection',
            'opt_out_mechanisms',
            'do_not_sell_detection',
            'privacy_policy_requirements',
            'consumer_rights_disclosure',
            'third_party_sharing'
        ]
    
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute CCPA compliance validation."""
        config = context.check_config
        
        violations = []
        
        # Check "Do Not Sell" mechanism
        dns_violations = self._check_do_not_sell_mechanism(context, config)
        violations.extend(dns_violations)
        
        # Check consumer rights disclosure
        rights_violations = self._check_consumer_rights_disclosure(context, config)
        violations.extend(rights_violations)
        
        # Check personal information collection disclosure
        collection_violations = self._check_personal_info_disclosure(context, config)
        violations.extend(collection_violations)
        
        # Check third-party data sharing
        sharing_violations = self._check_third_party_sharing(context, config)
        violations.extend(sharing_violations)
        
        passed = len(violations) == 0
        message = f"Found {len(violations)} CCPA compliance violations"
        
        return self._create_result(
            context=context,
            passed=passed,
            message=message,
            found_count=len(violations),
            expected_count=0,
            evidence=violations[:20]
        )
    
    def _check_do_not_sell_mechanism(
        self, 
        context: CheckContext, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for required 'Do Not Sell My Personal Information' mechanism."""
        violations = []
        
        # Look for "Do Not Sell" requests or links
        dns_requests = context.query.query().where_regex(
            'url', r'(do.*not.*sell|dns|ccpa|opt.*out)'
        ).requests()
        
        if not dns_requests:
            violations.append({
                'type': 'missing_do_not_sell_mechanism',
                'description': 'No "Do Not Sell" mechanism detected',
                'requirement': 'CCPA requires prominent Do Not Sell link'
            })
        
        # Check for Global Privacy Control (GPC) support
        gpc_support = self._check_gpc_support(context)
        if not gpc_support:
            violations.append({
                'type': 'missing_gpc_support',
                'description': 'Global Privacy Control signal not detected',
                'requirement': 'CCPA requires honoring GPC signals'
            })
        
        return violations
    
    def _check_consumer_rights_disclosure(
        self, 
        context: CheckContext, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check for consumer rights disclosure requirements."""
        violations = []
        
        rights_disclosure = config.get('consumer_rights_disclosure', {})
        
        # Look for privacy policy requests
        privacy_requests = context.query.query().where_regex(
            'url', r'(privacy.*policy|ccpa|consumer.*rights)'
        ).requests()
        
        if not privacy_requests:
            violations.append({
                'type': 'missing_privacy_policy',
                'description': 'No privacy policy detected',
                'requirement': 'CCPA requires comprehensive privacy policy'
            })
        
        # Check for required consumer rights information
        required_disclosures = rights_disclosure.get('required_rights', [
            'right_to_know', 'right_to_delete', 'right_to_opt_out',
            'right_to_non_discrimination'
        ])
        
        # This would need content analysis in a real implementation
        for right in required_disclosures:
            if not self._check_right_disclosure(context, right):
                violations.append({
                    'type': 'missing_consumer_right_disclosure',
                    'missing_right': right,
                    'requirement': f'CCPA requires disclosure of {right}'
                })
        
        return violations
    
    def _check_personal_info_disclosure(
        self, 
        context: CheckContext, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check personal information collection and use disclosure."""
        violations = []
        
        collection_config = config.get('personal_info_collection', {})
        
        # Analyze data collection patterns
        tracking_requests = self._identify_tracking_requests(context)
        
        if tracking_requests and not self._has_collection_disclosure(context):
            violations.append({
                'type': 'undisclosed_personal_info_collection',
                'tracking_requests_count': len(tracking_requests),
                'requirement': 'CCPA requires disclosure of personal info collection'
            })
        
        return violations
    
    def _check_third_party_sharing(
        self, 
        context: CheckContext, 
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Check third-party data sharing compliance."""
        violations = []
        
        sharing_config = config.get('third_party_sharing', {})
        
        # Identify potential third-party data sharing
        third_party_requests = self._identify_third_party_requests(context)
        
        for request in third_party_requests[:10]:  # Limit analysis
            if self._is_data_sharing_request(request):
                violations.append({
                    'type': 'potential_undisclosed_data_sharing',
                    'third_party_domain': urlparse(request.url).netloc,
                    'request_url': request.url,
                    'requirement': 'CCPA requires disclosure of data sharing'
                })
        
        return violations
    
    def _check_gpc_support(self, context: CheckContext) -> bool:
        """Check for Global Privacy Control support."""
        # Look for GPC-related requests or cookies
        gpc_requests = context.query.query().where_regex(
            'url', r'(gpc|global.*privacy.*control)'
        ).requests()
        
        gpc_headers = []
        for request in context.query.query().requests():
            if 'Sec-GPC' in request.request_headers:
                gpc_headers.append(request)
        
        return len(gpc_requests) > 0 or len(gpc_headers) > 0
    
    def _check_right_disclosure(self, context: CheckContext, right: str) -> bool:
        """Check if specific consumer right is disclosed (simplified)."""
        # In real implementation, would analyze page content
        return True  # Placeholder
    
    def _identify_tracking_requests(self, context: CheckContext) -> List[RequestLog]:
        """Identify requests that likely collect personal information."""
        tracking_patterns = [
            r'analytics?', r'tracking?', r'pixel', r'beacon',
            r'collect', r'data', r'metrics', r'stats'
        ]
        
        tracking_requests = []
        for pattern in tracking_patterns:
            requests = context.query.query().where_regex('url', pattern).requests()
            tracking_requests.extend(requests)
        
        return tracking_requests
    
    def _has_collection_disclosure(self, context: CheckContext) -> bool:
        """Check if personal information collection is disclosed."""
        # Look for privacy policy or collection notice
        disclosure_requests = context.query.query().where_regex(
            'url', r'(privacy|collection|notice|disclosure)'
        ).requests()
        
        return len(disclosure_requests) > 0
    
    def _identify_third_party_requests(self, context: CheckContext) -> List[RequestLog]:
        """Identify third-party requests that might involve data sharing."""
        requests = context.query.query().requests()
        
        # Get first-party domains
        first_party_domains = set()
        for page in context.indexes.pages.pages:
            first_party_domains.add(urlparse(page.url).netloc)
        
        # Filter to third-party requests
        third_party_requests = []
        for request in requests:
            request_domain = urlparse(request.url).netloc
            if request_domain not in first_party_domains:
                third_party_requests.append(request)
        
        return third_party_requests
    
    def _is_data_sharing_request(self, request: RequestLog) -> bool:
        """Determine if request likely involves data sharing."""
        # Check for data-sharing patterns
        sharing_patterns = [
            r'collect', r'track', r'analytics?', r'pixel',
            r'beacon', r'user.*data', r'profile'
        ]
        
        return any(re.search(pattern, request.url, re.IGNORECASE) 
                  for pattern in sharing_patterns)


@register_check("cookie_security")
class CookieSecurityCheck(BaseCheck):
    """Validate cookie security attributes and best practices."""
    
    def __init__(self, check_id: str, name: str, description: str):
        super().__init__(check_id, name, description)
    
    def get_supported_config_keys(self) -> List[str]:
        return [
            'require_secure_flag',
            'require_httponly_flag',
            'require_samesite_attribute',
            'secure_context_required',
            'cookie_prefix_validation',
            'max_cookie_age_days',
            'sensitive_cookie_patterns'
        ]
    
    def execute(self, context: CheckContext) -> CheckResult:
        """Execute cookie security validation."""
        config = context.check_config
        
        violations = []
        
        # Get all cookies
        cookies = context.query.query().cookies()
        
        for cookie in cookies:
            cookie_violations = self._validate_cookie_security(cookie, config, context)
            violations.extend(cookie_violations)
        
        passed = len(violations) == 0
        message = f"Found {len(violations)} cookie security violations across {len(cookies)} cookies"
        
        return self._create_result(
            context=context,
            passed=passed,
            message=message,
            found_count=len(violations),
            expected_count=0,
            evidence=violations[:20]
        )
    
    def _validate_cookie_security(
        self, 
        cookie: CookieRecord, 
        config: Dict[str, Any], 
        context: CheckContext
    ) -> List[Dict[str, Any]]:
        """Validate security attributes for a single cookie."""
        violations = []
        
        # Check Secure flag
        if config.get('require_secure_flag', True) and not cookie.secure:
            if self._is_https_context(context):
                violations.append({
                    'type': 'missing_secure_flag',
                    'cookie_name': cookie.name,
                    'cookie_domain': cookie.domain,
                    'risk': 'Cookie can be transmitted over unencrypted connections',
                    'recommendation': 'Add Secure flag to cookie attributes'
                })
        
        # Check HttpOnly flag
        sensitive_patterns = config.get('sensitive_cookie_patterns', [
            r'session', r'auth', r'login', r'token', r'csrf'
        ])
        
        if (config.get('require_httponly_flag', True) and 
            not cookie.http_only and 
            self._is_sensitive_cookie(cookie, sensitive_patterns)):
            
            violations.append({
                'type': 'missing_httponly_flag',
                'cookie_name': cookie.name,
                'cookie_domain': cookie.domain,
                'risk': 'Cookie accessible via JavaScript, vulnerable to XSS attacks',
                'recommendation': 'Add HttpOnly flag to prevent script access'
            })
        
        # Check SameSite attribute
        if config.get('require_samesite_attribute', True) and not cookie.same_site:
            violations.append({
                'type': 'missing_samesite_attribute',
                'cookie_name': cookie.name,
                'cookie_domain': cookie.domain,
                'risk': 'Cookie vulnerable to CSRF attacks',
                'recommendation': 'Add SameSite=Strict or SameSite=Lax attribute'
            })
        
        # Check cookie age limits
        max_age_days = config.get('max_cookie_age_days')
        if max_age_days and not cookie.is_session and cookie.expires:
            cookie_age_days = (cookie.expires - datetime.utcnow()).days
            if cookie_age_days > max_age_days:
                violations.append({
                    'type': 'excessive_cookie_age',
                    'cookie_name': cookie.name,
                    'cookie_age_days': cookie_age_days,
                    'max_allowed_days': max_age_days,
                    'risk': 'Long-lived cookies increase privacy and security risks'
                })
        
        # Check secure cookie prefixes
        if config.get('cookie_prefix_validation', True):
            prefix_violations = self._check_cookie_prefixes(cookie)
            violations.extend(prefix_violations)
        
        # Check for insecure cookie values
        value_violations = self._check_cookie_value_security(cookie)
        violations.extend(value_violations)
        
        return violations
    
    def _is_https_context(self, context: CheckContext) -> bool:
        """Determine if cookies are being set in HTTPS context."""
        for page in context.indexes.pages.pages:
            if page.url.startswith('https://'):
                return True
        return False
    
    def _is_sensitive_cookie(self, cookie: CookieRecord, patterns: List[str]) -> bool:
        """Determine if cookie contains sensitive information."""
        for pattern in patterns:
            if re.search(pattern, cookie.name, re.IGNORECASE):
                return True
        return False
    
    def _check_cookie_prefixes(self, cookie: CookieRecord) -> List[Dict[str, Any]]:
        """Validate secure cookie prefixes."""
        violations = []
        
        if cookie.name.startswith('__Secure-'):
            if not cookie.secure:
                violations.append({
                    'type': 'invalid_secure_prefix',
                    'cookie_name': cookie.name,
                    'issue': '__Secure- prefix requires Secure flag',
                    'risk': 'Prefix security guarantee violated'
                })
        
        elif cookie.name.startswith('__Host-'):
            if not cookie.secure:
                violations.append({
                    'type': 'invalid_host_prefix_secure',
                    'cookie_name': cookie.name,
                    'issue': '__Host- prefix requires Secure flag',
                    'risk': 'Prefix security guarantee violated'
                })
            
            if cookie.path != '/':
                violations.append({
                    'type': 'invalid_host_prefix_path',
                    'cookie_name': cookie.name,
                    'issue': '__Host- prefix requires Path=/',
                    'current_path': cookie.path,
                    'risk': 'Prefix security guarantee violated'
                })
            
            if '.' in cookie.domain:
                violations.append({
                    'type': 'invalid_host_prefix_domain',
                    'cookie_name': cookie.name,
                    'issue': '__Host- prefix requires no Domain attribute',
                    'current_domain': cookie.domain,
                    'risk': 'Prefix security guarantee violated'
                })
        
        return violations
    
    def _check_cookie_value_security(self, cookie: CookieRecord) -> List[Dict[str, Any]]:
        """Check cookie value for security issues."""
        violations = []
        
        if not cookie.value_redacted and cookie.value:
            # Check for potential sensitive data in cookie values
            sensitive_patterns = [
                (r'password', 'potential_password_in_cookie'),
                (r'token.*[a-fA-F0-9]{32,}', 'potential_token_in_cookie'),
                (r'key.*[a-fA-F0-9]{32,}', 'potential_key_in_cookie'),
                (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 'email_in_cookie')
            ]
            
            for pattern, violation_type in sensitive_patterns:
                if re.search(pattern, cookie.value, re.IGNORECASE):
                    violations.append({
                        'type': violation_type,
                        'cookie_name': cookie.name,
                        'risk': 'Sensitive data stored in cookie value',
                        'recommendation': 'Store sensitive data server-side with session reference'
                    })
        
        return violations