"""Cookie policy compliance engine for privacy regulations.

This module provides comprehensive cookie policy validation supporting multiple
privacy frameworks including GDPR, CCPA, and custom organizational policies.
Validates cookie attributes, consent requirements, and regulatory compliance.
"""

import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum
from urllib.parse import urlparse

from .models import CookieRecord, CookiePolicyIssue, PolicySeverity
from .config import PrivacyConfiguration, PolicyRule
from .classification import CookieClassifier, CookieCategory

logger = logging.getLogger(__name__)


class ComplianceFramework(str, Enum):
    """Supported privacy compliance frameworks."""
    GDPR = "gdpr"           # General Data Protection Regulation
    CCPA = "ccpa"           # California Consumer Privacy Act
    PIPEDA = "pipeda"       # Personal Information Protection and Electronic Documents Act
    LGPD = "lgpd"           # Lei Geral de Proteção de Dados
    CUSTOM = "custom"       # Custom organizational policies


class PolicyViolationType(str, Enum):
    """Types of policy violations."""
    SECURITY_ATTRIBUTE = "security_attribute"      # Missing security flags
    CONSENT_REQUIRED = "consent_required"          # Cookie requires consent
    RETENTION_PERIOD = "retention_period"          # Cookie exceeds retention limits
    PURPOSE_LIMITATION = "purpose_limitation"      # Cookie used beyond stated purpose
    DOMAIN_SCOPE = "domain_scope"                  # Invalid domain scope
    ESSENTIAL_CLASSIFICATION = "essential_classification"  # Incorrect essential status
    CROSS_BORDER_TRANSFER = "cross_border_transfer"        # International data transfer


class PolicyComplianceEngine:
    """Comprehensive cookie policy compliance validation engine.
    
    Validates cookies against multiple privacy frameworks and organizational
    policies, providing detailed compliance analysis and remediation guidance.
    """
    
    def __init__(self, config: Optional[PrivacyConfiguration] = None):
        """Initialize policy compliance engine.
        
        Args:
            config: Privacy configuration with policy rules
        """
        self.config = config
        self.classifier = CookieClassifier(config)
        
        # Rule evaluation functions
        self._rule_evaluators: Dict[str, Callable] = {
            'secure_flag_https': self._check_secure_flag,
            'same_site_required': self._check_same_site,
            'http_only_sessions': self._check_http_only,
            'analytics_consent_required': self._check_analytics_consent,
            'retention_period_limit': self._check_retention_period,
            'domain_scope_validation': self._check_domain_scope,
            'essential_classification': self._check_essential_classification,
        }
        
        # Framework-specific requirements
        self.framework_requirements = {
            ComplianceFramework.GDPR: {
                'consent_required_categories': [CookieCategory.ANALYTICS, CookieCategory.MARKETING, CookieCategory.SOCIAL],
                'max_retention_days': 365,
                'cross_border_restrictions': True,
                'explicit_consent_required': True,
            },
            ComplianceFramework.CCPA: {
                'consent_required_categories': [CookieCategory.MARKETING],
                'max_retention_days': 730,
                'cross_border_restrictions': False,
                'explicit_consent_required': False,  # Opt-out model
            },
            ComplianceFramework.PIPEDA: {
                'consent_required_categories': [CookieCategory.ANALYTICS, CookieCategory.MARKETING],
                'max_retention_days': 365,
                'cross_border_restrictions': True,
                'explicit_consent_required': True,
            },
        }
    
    def _check_secure_flag(self, cookie: CookieRecord, context: Dict[str, Any]) -> Optional[CookiePolicyIssue]:
        """Check if cookie has appropriate Secure flag.
        
        Args:
            cookie: Cookie to check
            context: Validation context (page_url, environment, etc.)
            
        Returns:
            Policy issue if violation found, None otherwise
        """
        page_url = context.get('page_url', '')
        environment = context.get('environment', 'production')
        
        if urlparse(page_url).scheme == 'https' and not cookie.secure:
            severity = PolicySeverity.HIGH if environment == 'production' else PolicySeverity.MEDIUM
            
            return CookiePolicyIssue(
                cookie_name=cookie.name,
                cookie_domain=cookie.domain,
                cookie_path=cookie.path,
                attribute="secure",
                expected="true",
                observed="false",
                severity=severity,
                rule_id="secure_flag_https",
                message=f"Cookie {cookie.name} lacks Secure flag on HTTPS site",
                scenario_id=context.get('scenario_id'),
                page_url=page_url,
                remediation="Add Secure attribute to cookies on HTTPS sites to prevent interception"
            )
        
        return None
    
    def _check_same_site(self, cookie: CookieRecord, context: Dict[str, Any]) -> Optional[CookiePolicyIssue]:
        """Check if cookie has appropriate SameSite attribute.
        
        Args:
            cookie: Cookie to check
            context: Validation context
            
        Returns:
            Policy issue if violation found, None otherwise
        """
        page_url = context.get('page_url', '')
        
        # Third-party cookies should have SameSite=None with Secure flag
        if not cookie.is_first_party:
            if not cookie.same_site or cookie.same_site.lower() != 'none':
                return CookiePolicyIssue(
                    cookie_name=cookie.name,
                    cookie_domain=cookie.domain,
                    cookie_path=cookie.path,
                    attribute="same_site",
                    expected="None",
                    observed=cookie.same_site or "not_set",
                    severity=PolicySeverity.MEDIUM,
                    rule_id="same_site_third_party",
                    message=f"Third-party cookie {cookie.name} should have SameSite=None",
                    page_url=page_url,
                    remediation="Set SameSite=None for third-party cookies (requires Secure flag)"
                )
            
            # Third-party cookies with SameSite=None must have Secure flag
            if cookie.same_site and cookie.same_site.lower() == 'none' and not cookie.secure:
                return CookiePolicyIssue(
                    cookie_name=cookie.name,
                    cookie_domain=cookie.domain,
                    cookie_path=cookie.path,
                    attribute="secure",
                    expected="true",
                    observed="false",
                    severity=PolicySeverity.HIGH,
                    rule_id="secure_required_same_site_none",
                    message=f"Cookie {cookie.name} with SameSite=None requires Secure flag",
                    page_url=page_url,
                    remediation="Add Secure flag to cookies with SameSite=None"
                )
        
        # First-party cookies should have SameSite protection
        else:
            if not cookie.same_site:
                expected_same_site = "Lax"  # Default for first-party cookies
                
                return CookiePolicyIssue(
                    cookie_name=cookie.name,
                    cookie_domain=cookie.domain,
                    cookie_path=cookie.path,
                    attribute="same_site",
                    expected=expected_same_site,
                    observed="not_set",
                    severity=PolicySeverity.MEDIUM,
                    rule_id="same_site_required",
                    message=f"First-party cookie {cookie.name} should have SameSite attribute",
                    page_url=page_url,
                    remediation="Set SameSite=Lax for most first-party cookies, or SameSite=Strict for sensitive cookies"
                )
        
        return None
    
    def _check_http_only(self, cookie: CookieRecord, context: Dict[str, Any]) -> Optional[CookiePolicyIssue]:
        """Check if session/authentication cookies have HttpOnly flag.
        
        Args:
            cookie: Cookie to check
            context: Validation context
            
        Returns:
            Policy issue if violation found, None otherwise
        """
        page_url = context.get('page_url', '')
        
        # Check if cookie should have HttpOnly flag
        should_be_http_only = (
            cookie.is_session or
            'session' in cookie.name.lower() or
            'jsessionid' in cookie.name.lower() or
            'phpsessid' in cookie.name.lower() or
            'asp.net_sessionid' in cookie.name.lower() or
            'auth' in cookie.name.lower() or
            'login' in cookie.name.lower() or
            'csrf' in cookie.name.lower() or
            'authenticity_token' in cookie.name.lower()
        )
        
        if should_be_http_only and not cookie.http_only:
            return CookiePolicyIssue(
                cookie_name=cookie.name,
                cookie_domain=cookie.domain,
                cookie_path=cookie.path,
                attribute="http_only",
                expected="true",
                observed="false",
                severity=PolicySeverity.HIGH,
                rule_id="http_only_sessions",
                message=f"Session/authentication cookie {cookie.name} should have HttpOnly flag",
                page_url=page_url,
                remediation="Add HttpOnly flag to prevent client-side script access to sensitive cookies"
            )
        
        return None
    
    def _check_analytics_consent(self, cookie: CookieRecord, context: Dict[str, Any]) -> Optional[CookiePolicyIssue]:
        """Check if analytics cookies require consent.
        
        Args:
            cookie: Cookie to check
            context: Validation context
            
        Returns:
            Policy issue if violation found, None otherwise
        """
        page_url = context.get('page_url', '')
        scenario_id = context.get('scenario_id', '')
        framework = context.get('framework', ComplianceFramework.GDPR)
        
        # Check if cookie is analytics/marketing and requires consent
        cookie_category = cookie.metadata.get('classification', {}).get('category') if cookie.metadata else None
        
        framework_reqs = self.framework_requirements.get(framework, {})
        consent_required_categories = framework_reqs.get('consent_required_categories', [])
        
        if (cookie_category in [cat.value for cat in consent_required_categories] and 
            not cookie.essential):
            
            # Check if we're in a privacy scenario where this cookie should be absent
            violation_scenarios = ['gpc_on', 'cmp_reject_all']
            
            if scenario_id in violation_scenarios:
                return CookiePolicyIssue(
                    cookie_name=cookie.name,
                    cookie_domain=cookie.domain,
                    cookie_path=cookie.path,
                    attribute="consent_compliance",
                    expected="absent",
                    observed="present",
                    severity=PolicySeverity.CRITICAL,
                    rule_id="analytics_consent_required",
                    message=f"{cookie_category.title()} cookie {cookie.name} should not be present when consent is rejected or GPC is enabled",
                    scenario_id=scenario_id,
                    page_url=page_url,
                    remediation=f"Remove {cookie_category} cookies when user rejects consent or privacy signals are present"
                )
            
            # In baseline scenario, flag for consent requirement
            elif scenario_id == 'baseline' or not scenario_id:
                return CookiePolicyIssue(
                    cookie_name=cookie.name,
                    cookie_domain=cookie.domain,
                    cookie_path=cookie.path,
                    attribute="consent",
                    expected="required",
                    observed="not_verified",
                    severity=PolicySeverity.MEDIUM,
                    rule_id="analytics_consent_required",
                    message=f"{cookie_category.title()} cookie {cookie.name} requires user consent under {framework.value.upper()}",
                    scenario_id=scenario_id,
                    page_url=page_url,
                    remediation=f"Implement consent management for {cookie_category} cookies per {framework.value.upper()} requirements"
                )
        
        return None
    
    def _check_retention_period(self, cookie: CookieRecord, context: Dict[str, Any]) -> Optional[CookiePolicyIssue]:
        """Check if cookie retention period complies with policy.
        
        Args:
            cookie: Cookie to check
            context: Validation context
            
        Returns:
            Policy issue if violation found, None otherwise
        """
        page_url = context.get('page_url', '')
        framework = context.get('framework', ComplianceFramework.GDPR)
        
        if not cookie.expires:
            # Session cookies are generally acceptable
            return None
        
        framework_reqs = self.framework_requirements.get(framework, {})
        max_retention_days = framework_reqs.get('max_retention_days', 365)
        
        # Calculate retention period
        now = datetime.utcnow()
        retention_days = (cookie.expires - now).days
        
        # Check for non-essential cookies with long retention
        if retention_days > max_retention_days and not cookie.essential:
            return CookiePolicyIssue(
                cookie_name=cookie.name,
                cookie_domain=cookie.domain,
                cookie_path=cookie.path,
                attribute="retention_period",
                expected=f"<= {max_retention_days} days",
                observed=f"{retention_days} days",
                severity=PolicySeverity.MEDIUM,
                rule_id="retention_period_limit",
                message=f"Cookie {cookie.name} retention period exceeds {framework.value.upper()} guidelines",
                page_url=page_url,
                remediation=f"Reduce cookie retention to {max_retention_days} days or less for non-essential cookies"
            )
        
        return None
    
    def _check_domain_scope(self, cookie: CookieRecord, context: Dict[str, Any]) -> Optional[CookiePolicyIssue]:
        """Check if cookie domain scope is appropriate.
        
        Args:
            cookie: Cookie to check
            context: Validation context
            
        Returns:
            Policy issue if violation found, None otherwise
        """
        page_url = context.get('page_url', '')
        page_domain = urlparse(page_url).netloc
        
        # Check for overly broad domain scope
        cookie_domain = cookie.domain.lstrip('.')
        
        # Warning for domain cookies that are too broad
        if cookie.domain.startswith('.') and cookie_domain != page_domain:
            # Check if it's a reasonable parent domain
            page_etld1 = self.classifier._extract_etld_plus_one(page_domain)
            cookie_etld1 = self.classifier._extract_etld_plus_one(cookie_domain)
            
            if cookie_etld1 != page_etld1:
                return CookiePolicyIssue(
                    cookie_name=cookie.name,
                    cookie_domain=cookie.domain,
                    cookie_path=cookie.path,
                    attribute="domain_scope",
                    expected=f"Same eTLD+1 as {page_domain}",
                    observed=cookie.domain,
                    severity=PolicySeverity.LOW,
                    rule_id="domain_scope_validation",
                    message=f"Cookie {cookie.name} has overly broad domain scope",
                    page_url=page_url,
                    remediation="Limit cookie domain to the specific domain or eTLD+1 needed"
                )
        
        return None
    
    def _check_essential_classification(self, cookie: CookieRecord, context: Dict[str, Any]) -> Optional[CookiePolicyIssue]:
        """Check if cookie's essential classification is accurate.
        
        Args:
            cookie: Cookie to check
            context: Validation context
            
        Returns:
            Policy issue if violation found, None otherwise
        """
        page_url = context.get('page_url', '')
        scenario_id = context.get('scenario_id', '')
        
        # If cookie claims to be essential but is absent in privacy scenarios, flag it
        if (cookie.essential is True and 
            scenario_id in ['gpc_on', 'cmp_reject_all']):
            
            # Essential cookies should still be present even with privacy signals
            # If we're evaluating a cookie that's missing in a privacy scenario,
            # this would be called differently. This check is for present cookies.
            
            # Check if cookie looks like analytics/marketing but claims to be essential
            cookie_category = cookie.metadata.get('classification', {}).get('category') if cookie.metadata else None
            
            if cookie_category in [CookieCategory.ANALYTICS.value, CookieCategory.MARKETING.value]:
                return CookiePolicyIssue(
                    cookie_name=cookie.name,
                    cookie_domain=cookie.domain,
                    cookie_path=cookie.path,
                    attribute="essential_classification",
                    expected="false",
                    observed="true",
                    severity=PolicySeverity.HIGH,
                    rule_id="essential_classification",
                    message=f"{cookie_category.title()} cookie {cookie.name} incorrectly classified as essential",
                    scenario_id=scenario_id,
                    page_url=page_url,
                    remediation=f"Review essential classification - {cookie_category} cookies typically require consent"
                )
        
        return None
    
    def validate_cookie_policy(
        self, 
        cookies: List[CookieRecord], 
        page_url: str,
        framework: ComplianceFramework = ComplianceFramework.GDPR,
        scenario_id: Optional[str] = None,
        environment: str = "production"
    ) -> List[CookiePolicyIssue]:
        """Validate cookies against privacy policy requirements.
        
        Args:
            cookies: List of cookies to validate
            page_url: URL where cookies were found
            framework: Privacy framework to validate against
            scenario_id: Current scenario identifier
            environment: Environment context
            
        Returns:
            List of policy violations found
        """
        issues = []
        
        # Create validation context
        context = {
            'page_url': page_url,
            'framework': framework,
            'scenario_id': scenario_id,
            'environment': environment,
            'page_domain': urlparse(page_url).netloc,
        }
        
        for cookie in cookies:
            # Run all applicable rule evaluators
            for rule_id, evaluator in self._rule_evaluators.items():
                try:
                    issue = evaluator(cookie, context)
                    if issue:
                        issues.append(issue)
                except Exception as e:
                    logger.warning(f"Error evaluating rule {rule_id} for cookie {cookie.name}: {e}")
        
        # Run configuration-based policy rules if available
        if self.config and self.config.policy_rules:
            issues.extend(self._evaluate_config_rules(cookies, context))
        
        logger.info(f"Found {len(issues)} policy issues for {len(cookies)} cookies under {framework.value}")
        
        return issues
    
    def _evaluate_config_rules(self, cookies: List[CookieRecord], context: Dict[str, Any]) -> List[CookiePolicyIssue]:
        """Evaluate configuration-based policy rules.
        
        Args:
            cookies: List of cookies to validate
            context: Validation context
            
        Returns:
            List of policy violations found
        """
        issues = []
        
        for rule in self.config.policy_rules:
            try:
                for cookie in cookies:
                    issue = self._evaluate_rule(rule, cookie, context)
                    if issue:
                        issues.append(issue)
            except Exception as e:
                logger.warning(f"Error evaluating config rule {rule.id}: {e}")
        
        return issues
    
    def _evaluate_rule(self, rule: PolicyRule, cookie: CookieRecord, context: Dict[str, Any]) -> Optional[CookiePolicyIssue]:
        """Evaluate a single policy rule against a cookie.
        
        Args:
            rule: Policy rule to evaluate
            cookie: Cookie to check
            context: Validation context
            
        Returns:
            Policy issue if rule violated, None otherwise
        """
        # Simple rule evaluation - in production would use a proper expression evaluator
        condition = rule.condition
        
        # Replace placeholders in condition
        condition = condition.replace('cookie.secure', str(cookie.secure).lower())
        condition = condition.replace('cookie.http_only', str(cookie.http_only).lower())
        condition = condition.replace('cookie.same_site', f"'{cookie.same_site or ''}'")
        condition = condition.replace('cookie.name', f"'{cookie.name}'")
        condition = condition.replace('cookie.domain', f"'{cookie.domain}'")
        condition = condition.replace('cookie.is_session', str(cookie.is_session).lower())
        condition = condition.replace('cookie.is_first_party', str(cookie.is_first_party).lower())
        condition = condition.replace('cookie.essential', str(cookie.essential).lower())
        
        # Context replacements
        condition = condition.replace('site_protocol', f"'{urlparse(context['page_url']).scheme}'")
        condition = condition.replace('scenario', f"'{context.get('scenario_id', '')}'")
        condition = condition.replace('environment', f"'{context.get('environment', '')}'")
        
        # Cookie category and analytics classification
        cookie_category = cookie.metadata.get('classification', {}).get('category', '') if cookie.metadata else ''
        condition = condition.replace('cookie.category', f"'{cookie_category}'")
        
        is_analytics = (cookie.metadata.get('classification', {}).get('is_analytics_heuristic', False) 
                       if cookie.metadata else False)
        condition = condition.replace('cookie.is_analytics', str(is_analytics).lower())
        
        # Simple pattern matching for "matches" operator
        if 'matches' in condition:
            # Extract pattern matching expressions
            import re as regex_module
            match_patterns = regex_module.findall(r"'([^']+)'\s+matches\s+'([^']+)'", condition)
            for text, pattern in match_patterns:
                try:
                    matches = bool(regex_module.match(pattern, text))
                    condition = condition.replace(f"'{text}' matches '{pattern}'", str(matches).lower())
                except regex_module.error:
                    matches = False
                    condition = condition.replace(f"'{text}' matches '{pattern}'", str(matches).lower())
        
        try:
            # Safely evaluate the condition using a restricted evaluator
            condition_result = self._safe_eval(condition)
            
            if condition_result:
                return CookiePolicyIssue(
                    cookie_name=cookie.name,
                    cookie_domain=cookie.domain,
                    cookie_path=cookie.path,
                    attribute=rule.attribute,
                    expected=rule.expected,
                    observed="condition_met",
                    severity=PolicySeverity(rule.severity),
                    rule_id=rule.id,
                    message=rule.description,
                    scenario_id=context.get('scenario_id'),
                    page_url=context['page_url'],
                    remediation=rule.remediation
                )
        except Exception as e:
            logger.warning(f"Error evaluating rule condition '{condition}': {e}")
        
        return None
    
    def generate_compliance_report(
        self, 
        cookies: List[CookieRecord], 
        issues: List[CookiePolicyIssue],
        page_url: str,
        framework: ComplianceFramework = ComplianceFramework.GDPR
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report.
        
        Args:
            cookies: List of analyzed cookies
            issues: List of policy violations
            page_url: URL analyzed
            framework: Privacy framework used
            
        Returns:
            Detailed compliance analysis report
        """
        # Group issues by severity
        issues_by_severity = {}
        for issue in issues:
            severity = issue.severity
            if severity not in issues_by_severity:
                issues_by_severity[severity] = []
            issues_by_severity[severity].append(issue)
        
        # Group issues by type
        issues_by_type = {}
        for issue in issues:
            issue_type = self._determine_issue_type(issue)
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(issue)
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(cookies, issues, framework)
        
        # Framework-specific analysis
        framework_analysis = self._analyze_framework_compliance(cookies, issues, framework)
        
        return {
            'compliance_summary': {
                'total_cookies': len(cookies),
                'total_issues': len(issues),
                'compliance_score': compliance_score,
                'framework': framework.value,
                'page_url': page_url,
                'analysis_time': datetime.utcnow().isoformat(),
            },
            'issues_by_severity': {
                severity: len(issue_list) for severity, issue_list in issues_by_severity.items()
            },
            'issues_by_type': {
                issue_type: len(issue_list) for issue_type, issue_list in issues_by_type.items()
            },
            'framework_compliance': framework_analysis,
            'recommendations': self._generate_compliance_recommendations(issues, framework),
            'detailed_issues': [
                {
                    'rule_id': issue.rule_id,
                    'cookie': f"{issue.cookie_name}@{issue.cookie_domain}",
                    'severity': issue.severity,
                    'message': issue.message,
                    'remediation': issue.remediation,
                } for issue in issues
            ]
        }
    
    def _determine_issue_type(self, issue: CookiePolicyIssue) -> PolicyViolationType:
        """Determine the type of policy violation.
        
        Args:
            issue: Policy issue to classify
            
        Returns:
            Policy violation type
        """
        if issue.attribute in ['secure', 'http_only', 'same_site']:
            return PolicyViolationType.SECURITY_ATTRIBUTE
        elif 'consent' in issue.attribute:
            return PolicyViolationType.CONSENT_REQUIRED
        elif 'retention' in issue.rule_id:
            return PolicyViolationType.RETENTION_PERIOD
        elif 'domain' in issue.rule_id:
            return PolicyViolationType.DOMAIN_SCOPE
        elif 'essential' in issue.rule_id:
            return PolicyViolationType.ESSENTIAL_CLASSIFICATION
        else:
            return PolicyViolationType.SECURITY_ATTRIBUTE  # Default
    
    def _calculate_compliance_score(
        self, 
        cookies: List[CookieRecord], 
        issues: List[CookiePolicyIssue], 
        framework: ComplianceFramework
    ) -> float:
        """Calculate overall compliance score.
        
        Args:
            cookies: List of cookies analyzed
            issues: List of policy violations
            framework: Privacy framework
            
        Returns:
            Compliance score (0-100)
        """
        if not cookies:
            return 100.0
        
        # Weight violations by severity
        severity_weights = {
            PolicySeverity.CRITICAL: 10,
            PolicySeverity.HIGH: 5,
            PolicySeverity.MEDIUM: 2,
            PolicySeverity.LOW: 1,
        }
        
        total_deductions = sum(severity_weights.get(issue.severity, 1) for issue in issues)
        max_possible_deductions = len(cookies) * severity_weights[PolicySeverity.CRITICAL]
        
        if max_possible_deductions == 0:
            return 100.0
        
        # Calculate score (0-100)
        score = max(0, 100 - (total_deductions / max_possible_deductions * 100))
        
        return round(score, 2)
    
    def _analyze_framework_compliance(
        self, 
        cookies: List[CookieRecord], 
        issues: List[CookiePolicyIssue], 
        framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """Analyze compliance with specific framework requirements.
        
        Args:
            cookies: List of cookies analyzed
            issues: List of policy violations
            framework: Privacy framework
            
        Returns:
            Framework-specific compliance analysis
        """
        framework_reqs = self.framework_requirements.get(framework, {})
        
        # Analyze consent requirements
        consent_required_categories = framework_reqs.get('consent_required_categories', [])
        
        consent_analysis = {
            'cookies_requiring_consent': 0,
            'consent_violations': 0,
        }
        
        for cookie in cookies:
            cookie_category = cookie.metadata.get('classification', {}).get('category') if cookie.metadata else None
            
            if cookie_category in [cat.value for cat in consent_required_categories] and not cookie.essential:
                consent_analysis['cookies_requiring_consent'] += 1
                
                # Check if there are consent-related violations for this cookie
                if any(issue.cookie_name == cookie.name and 'consent' in issue.attribute 
                      for issue in issues):
                    consent_analysis['consent_violations'] += 1
        
        return {
            'framework': framework.value,
            'consent_analysis': consent_analysis,
            'security_compliance': {
                'secure_flag_issues': len([i for i in issues if i.attribute == 'secure']),
                'http_only_issues': len([i for i in issues if i.attribute == 'http_only']),
                'same_site_issues': len([i for i in issues if i.attribute == 'same_site']),
            },
            'retention_compliance': {
                'max_retention_days': framework_reqs.get('max_retention_days', 365),
                'retention_violations': len([i for i in issues if 'retention' in i.rule_id]),
            }
        }
    
    def _generate_compliance_recommendations(
        self, 
        issues: List[CookiePolicyIssue], 
        framework: ComplianceFramework
    ) -> List[str]:
        """Generate compliance improvement recommendations.
        
        Args:
            issues: List of policy violations
            framework: Privacy framework
            
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        
        # Group by issue type for recommendations
        security_issues = [i for i in issues if i.attribute in ['secure', 'http_only', 'same_site']]
        consent_issues = [i for i in issues if 'consent' in i.attribute]
        retention_issues = [i for i in issues if 'retention' in i.rule_id]
        
        if security_issues:
            recommendations.append(
                f"Address {len(security_issues)} cookie security issues: "
                "implement proper Secure, HttpOnly, and SameSite attributes"
            )
        
        if consent_issues:
            recommendations.append(
                f"Implement consent management for {len(consent_issues)} cookies "
                f"to comply with {framework.value.upper()} requirements"
            )
        
        if retention_issues:
            recommendations.append(
                f"Review retention periods for {len(retention_issues)} cookies "
                f"to meet {framework.value.upper()} data minimization requirements"
            )
        
        # Critical issues get priority recommendations
        critical_issues = [i for i in issues if i.severity == PolicySeverity.CRITICAL]
        if critical_issues:
            recommendations.insert(0, 
                f"URGENT: Address {len(critical_issues)} critical privacy violations "
                "that may result in regulatory non-compliance"
            )
        
        return recommendations

    def _safe_eval(self, expression: str) -> bool:
        """Safely evaluate a boolean expression with restricted operations.

        Args:
            expression: String expression to evaluate

        Returns:
            Boolean result of evaluation

        Raises:
            ValueError: If expression contains unsafe operations
        """
        import ast
        import operator

        # Define allowed operations
        allowed_ops = {
            ast.And: operator.and_,
            ast.Or: operator.or_,
            ast.Not: operator.not_,
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.In: lambda x, y: x in y,
            ast.NotIn: lambda x, y: x not in y,
        }

        # Define allowed literals
        allowed_names = {'true', 'false', 'True', 'False'}

        def _eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Name):
                if node.id.lower() in ['true', 'false']:
                    return node.id.lower() == 'true'
                else:
                    raise ValueError(f"Unknown identifier: {node.id}")
            elif isinstance(node, ast.BoolOp):
                op = allowed_ops.get(type(node.op))
                if not op:
                    raise ValueError(f"Unsupported boolean operator: {type(node.op)}")
                values = [_eval_node(value) for value in node.values]
                result = values[0]
                for value in values[1:]:
                    result = op(result, value)
                return result
            elif isinstance(node, ast.UnaryOp):
                op = allowed_ops.get(type(node.op))
                if not op:
                    raise ValueError(f"Unsupported unary operator: {type(node.op)}")
                return op(_eval_node(node.operand))
            elif isinstance(node, ast.Compare):
                left = _eval_node(node.left)
                for op, comparator in zip(node.ops, node.comparators):
                    op_func = allowed_ops.get(type(op))
                    if not op_func:
                        raise ValueError(f"Unsupported comparison operator: {type(op)}")
                    right = _eval_node(comparator)
                    if not op_func(left, right):
                        return False
                    left = right
                return True
            else:
                raise ValueError(f"Unsupported AST node type: {type(node)}")

        try:
            # Parse the expression
            tree = ast.parse(expression, mode='eval')
            return _eval_node(tree.body)
        except (SyntaxError, ValueError) as e:
            logger.warning(f"Failed to safely evaluate expression '{expression}': {e}")
            return False