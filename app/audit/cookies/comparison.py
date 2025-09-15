"""Cross-scenario differential analysis for privacy testing.

This module provides sophisticated comparison capabilities between different
privacy scenarios, analyzing cookie differences, policy compliance changes,
and privacy signal effectiveness across scenarios.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
from enum import Enum

from .models import CookieRecord, CookieDiff, CookieChange, CookieChangeType, ScenarioCookieReport, CookiePolicyIssue

logger = logging.getLogger(__name__)


class ComparisonMetric(str, Enum):
    """Metrics for scenario comparison."""
    COOKIE_COUNT = "cookie_count"
    THIRD_PARTY_COOKIES = "third_party_cookies"
    ANALYTICS_COOKIES = "analytics_cookies"
    MARKETING_COOKIES = "marketing_cookies"
    NON_ESSENTIAL_COOKIES = "non_essential_cookies"
    POLICY_VIOLATIONS = "policy_violations"
    SECURITY_ISSUES = "security_issues"


class ScenarioComparator:
    """Advanced comparison engine for analyzing differences between privacy scenarios.
    
    Provides detailed analysis of cookie changes, policy improvements, and
    privacy signal effectiveness across different testing scenarios.
    """
    
    def __init__(self):
        """Initialize scenario comparator."""
        self.comparison_cache: Dict[str, CookieDiff] = {}
    
    def _generate_cookie_key(self, cookie: CookieRecord) -> str:
        """Generate unique key for cookie identification.
        
        Args:
            cookie: Cookie record
            
        Returns:
            Unique cookie identifier
        """
        return f"{cookie.name}@{cookie.domain}{cookie.path}"
    
    def compare_scenarios(
        self, 
        baseline_report: ScenarioCookieReport,
        variant_report: ScenarioCookieReport
    ) -> CookieDiff:
        """Compare two scenario reports and generate detailed diff.
        
        Args:
            baseline_report: Baseline scenario report
            variant_report: Variant scenario report
            
        Returns:
            Detailed cookie difference analysis
        """
        comparison_key = f"{baseline_report.scenario_id}_vs_{variant_report.scenario_id}"
        
        if comparison_key in self.comparison_cache:
            return self.comparison_cache[comparison_key]
        
        # Create cookie lookup maps
        baseline_cookies = {
            self._generate_cookie_key(c): c for c in baseline_report.cookies
        }
        variant_cookies = {
            self._generate_cookie_key(c): c for c in variant_report.cookies
        }
        
        # Initialize diff object
        diff = CookieDiff(
            baseline_scenario=baseline_report.scenario_id,
            variant_scenario=variant_report.scenario_id,
            page_url=baseline_report.page_url
        )
        
        # Find added cookies (in variant but not baseline)
        for key, cookie in variant_cookies.items():
            if key not in baseline_cookies:
                diff.added_cookies.append(cookie)
        
        # Find removed cookies (in baseline but not variant)
        for key, cookie in baseline_cookies.items():
            if key not in variant_cookies:
                diff.removed_cookies.append(cookie)
        
        # Find modified cookies (present in both but changed)
        for key in baseline_cookies:
            if key in variant_cookies:
                baseline_cookie = baseline_cookies[key]
                variant_cookie = variant_cookies[key]
                
                # Check for attribute changes
                changes = self._detect_cookie_changes(baseline_cookie, variant_cookie)
                if changes:
                    cookie_change = CookieChange(
                        cookie_key=key,
                        change_type=CookieChangeType.MODIFIED,
                        cookie_name=baseline_cookie.name,
                        cookie_domain=baseline_cookie.domain,
                        cookie_path=baseline_cookie.path,
                        attribute_changes=changes,
                        baseline_value=self._cookie_to_dict(baseline_cookie),
                        variant_value=self._cookie_to_dict(variant_cookie)
                    )
                    diff.modified_cookies.append(cookie_change)
                else:
                    # Cookie unchanged
                    diff.unchanged_cookies.append(baseline_cookie)
        
        # Calculate reduction percentage
        baseline_count = len(baseline_report.cookies)
        variant_count = len(variant_report.cookies)
        
        if baseline_count > 0:
            diff.reduction_percentage = ((baseline_count - variant_count) / baseline_count) * 100
        
        # Analyze policy impact
        diff.violations_resolved = self._count_violations_resolved(
            baseline_report.policy_issues, variant_report.policy_issues
        )
        diff.violations_introduced = self._count_violations_introduced(
            baseline_report.policy_issues, variant_report.policy_issues
        )
        diff.policy_improvement = diff.violations_resolved > diff.violations_introduced
        
        # Update total changes count
        diff.total_changes = len(diff.added_cookies) + len(diff.removed_cookies) + len(diff.modified_cookies)
        
        # Cache result
        self.comparison_cache[comparison_key] = diff
        
        logger.info(f"Scenario comparison complete: {baseline_report.scenario_id} vs {variant_report.scenario_id}")
        logger.info(f"Changes: +{len(diff.added_cookies)} -{len(diff.removed_cookies)} ~{len(diff.modified_cookies)}")
        
        return diff
    
    def _detect_cookie_changes(
        self, 
        baseline_cookie: CookieRecord, 
        variant_cookie: CookieRecord
    ) -> Dict[str, Dict[str, Any]]:
        """Detect specific attribute changes between two cookies.
        
        Args:
            baseline_cookie: Original cookie
            variant_cookie: Modified cookie
            
        Returns:
            Dictionary of attribute changes
        """
        changes = {}
        
        # Compare key attributes
        attributes_to_check = [
            'value', 'expires', 'max_age', 'secure', 'http_only', 
            'same_site', 'size', 'essential'
        ]
        
        for attr in attributes_to_check:
            baseline_val = getattr(baseline_cookie, attr, None)
            variant_val = getattr(variant_cookie, attr, None)
            
            if baseline_val != variant_val:
                # Handle special cases for comparison
                if attr == 'expires' and baseline_val and variant_val:
                    # Consider cookies with similar expiration times as unchanged
                    time_diff = abs((baseline_val - variant_val).total_seconds())
                    if time_diff < 60:  # Less than 1 minute difference
                        continue
                
                changes[attr] = {
                    'from': baseline_val,
                    'to': variant_val
                }
        
        return changes
    
    def _cookie_to_dict(self, cookie: CookieRecord) -> Dict[str, Any]:
        """Convert cookie to dictionary for comparison.
        
        Args:
            cookie: Cookie record
            
        Returns:
            Cookie as dictionary
        """
        return {
            'name': cookie.name,
            'value': cookie.value,
            'domain': cookie.domain,
            'path': cookie.path,
            'expires': cookie.expires.isoformat() if cookie.expires else None,
            'secure': cookie.secure,
            'http_only': cookie.http_only,
            'same_site': cookie.same_site,
            'essential': cookie.essential,
        }
    
    def _count_violations_resolved(
        self, 
        baseline_issues: List[CookiePolicyIssue],
        variant_issues: List[CookiePolicyIssue]
    ) -> int:
        """Count policy violations resolved in variant scenario.
        
        Args:
            baseline_issues: Issues in baseline scenario
            variant_issues: Issues in variant scenario
            
        Returns:
            Number of resolved violations
        """
        baseline_issue_keys = {
            f"{issue.cookie_name}@{issue.cookie_domain}:{issue.rule_id}"
            for issue in baseline_issues
        }
        
        variant_issue_keys = {
            f"{issue.cookie_name}@{issue.cookie_domain}:{issue.rule_id}"
            for issue in variant_issues
        }
        
        resolved = baseline_issue_keys - variant_issue_keys
        return len(resolved)
    
    def _count_violations_introduced(
        self, 
        baseline_issues: List[CookiePolicyIssue],
        variant_issues: List[CookiePolicyIssue]
    ) -> int:
        """Count new policy violations introduced in variant scenario.
        
        Args:
            baseline_issues: Issues in baseline scenario
            variant_issues: Issues in variant scenario
            
        Returns:
            Number of new violations
        """
        baseline_issue_keys = {
            f"{issue.cookie_name}@{issue.cookie_domain}:{issue.rule_id}"
            for issue in baseline_issues
        }
        
        variant_issue_keys = {
            f"{issue.cookie_name}@{issue.cookie_domain}:{issue.rule_id}"
            for issue in variant_issues
        }
        
        introduced = variant_issue_keys - baseline_issue_keys
        return len(introduced)
    
    def compare_multiple_scenarios(
        self, 
        scenario_reports: Dict[str, ScenarioCookieReport],
        baseline_scenario_id: str = 'baseline'
    ) -> Dict[str, CookieDiff]:
        """Compare multiple scenarios against a baseline.
        
        Args:
            scenario_reports: Dictionary of scenario reports
            baseline_scenario_id: ID of baseline scenario
            
        Returns:
            Dictionary mapping scenario IDs to their diffs vs baseline
        """
        if baseline_scenario_id not in scenario_reports:
            logger.error(f"Baseline scenario '{baseline_scenario_id}' not found")
            return {}
        
        baseline_report = scenario_reports[baseline_scenario_id]
        diffs = {}
        
        for scenario_id, report in scenario_reports.items():
            if scenario_id != baseline_scenario_id:
                diff = self.compare_scenarios(baseline_report, report)
                diffs[scenario_id] = diff
        
        return diffs
    
    def analyze_privacy_signal_effectiveness(
        self, 
        scenario_reports: Dict[str, ScenarioCookieReport]
    ) -> Dict[str, Any]:
        """Analyze effectiveness of privacy signals across scenarios.
        
        Args:
            scenario_reports: Dictionary of scenario reports
            
        Returns:
            Privacy signal effectiveness analysis
        """
        baseline_report = scenario_reports.get('baseline')
        if not baseline_report:
            return {'error': 'No baseline scenario found for comparison'}
        
        analysis = {
            'baseline_cookies': len(baseline_report.cookies),
            'signal_effectiveness': {},
            'overall_assessment': {}
        }
        
        # Analyze GPC effectiveness
        gpc_report = scenario_reports.get('gpc_on')
        if gpc_report:
            gpc_diff = self.compare_scenarios(baseline_report, gpc_report)
            
            effectiveness = self._calculate_signal_effectiveness(
                baseline_report, gpc_report, gpc_diff
            )
            
            analysis['signal_effectiveness']['gpc'] = {
                'cookie_reduction': gpc_diff.reduction_percentage,
                'cookies_removed': len(gpc_diff.removed_cookies),
                'non_essential_removed': len([
                    c for c in gpc_diff.removed_cookies 
                    if c.essential is False
                ]),
                'analytics_removed': len([
                    c for c in gpc_diff.removed_cookies
                    if c.metadata and c.metadata.get('classification', {}).get('category') == 'analytics'
                ]),
                'effectiveness_score': effectiveness,
                'assessment': self._assess_effectiveness(effectiveness)
            }
        
        # Analyze CMP effectiveness
        cmp_reject_report = scenario_reports.get('cmp_reject_all')
        if cmp_reject_report:
            cmp_diff = self.compare_scenarios(baseline_report, cmp_reject_report)
            
            effectiveness = self._calculate_signal_effectiveness(
                baseline_report, cmp_reject_report, cmp_diff
            )
            
            analysis['signal_effectiveness']['cmp_reject'] = {
                'cookie_reduction': cmp_diff.reduction_percentage,
                'cookies_removed': len(cmp_diff.removed_cookies),
                'non_essential_removed': len([
                    c for c in cmp_diff.removed_cookies 
                    if c.essential is False
                ]),
                'effectiveness_score': effectiveness,
                'assessment': self._assess_effectiveness(effectiveness)
            }
        
        # Overall privacy assessment
        analysis['overall_assessment'] = self._generate_overall_assessment(
            scenario_reports, analysis['signal_effectiveness']
        )
        
        return analysis
    
    def _calculate_signal_effectiveness(
        self,
        baseline_report: ScenarioCookieReport,
        variant_report: ScenarioCookieReport,
        diff: CookieDiff
    ) -> float:
        """Calculate effectiveness score for a privacy signal.
        
        Args:
            baseline_report: Baseline scenario
            variant_report: Variant scenario with privacy signal
            diff: Scenario difference
            
        Returns:
            Effectiveness score (0-100)
        """
        score = 0
        
        # Cookie reduction (40 points)
        if diff.reduction_percentage > 0:
            score += min(40, diff.reduction_percentage * 0.8)
        
        # Non-essential cookie removal (30 points)
        non_essential_baseline = len([
            c for c in baseline_report.cookies if c.essential is False
        ])
        non_essential_removed = len([
            c for c in diff.removed_cookies if c.essential is False
        ])
        
        if non_essential_baseline > 0:
            non_essential_reduction = (non_essential_removed / non_essential_baseline) * 100
            score += min(30, non_essential_reduction * 0.3)
        
        # Third-party cookie reduction (20 points)
        third_party_baseline = len([
            c for c in baseline_report.cookies if not c.is_first_party
        ])
        third_party_removed = len([
            c for c in diff.removed_cookies if not c.is_first_party
        ])
        
        if third_party_baseline > 0:
            third_party_reduction = (third_party_removed / third_party_baseline) * 100
            score += min(20, third_party_reduction * 0.2)
        
        # Policy improvement (10 points)
        if diff.policy_improvement:
            score += 10
        
        return min(100, score)
    
    def _assess_effectiveness(self, score: float) -> str:
        """Assess effectiveness based on score.
        
        Args:
            score: Effectiveness score
            
        Returns:
            Effectiveness assessment string
        """
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "moderate"
        elif score >= 20:
            return "poor"
        else:
            return "ineffective"
    
    def _generate_overall_assessment(
        self,
        scenario_reports: Dict[str, ScenarioCookieReport],
        signal_effectiveness: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate overall privacy assessment.
        
        Args:
            scenario_reports: All scenario reports
            signal_effectiveness: Signal effectiveness analysis
            
        Returns:
            Overall assessment
        """
        assessment = {
            'privacy_score': 0,
            'compliance_level': 'unknown',
            'recommendations': []
        }
        
        # Calculate overall privacy score
        total_signals = len(signal_effectiveness)
        if total_signals > 0:
            avg_effectiveness = sum(
                data.get('effectiveness_score', 0)
                for data in signal_effectiveness.values()
            ) / total_signals
            
            assessment['privacy_score'] = avg_effectiveness
        
        # Determine compliance level
        if assessment['privacy_score'] >= 80:
            assessment['compliance_level'] = 'excellent'
        elif assessment['privacy_score'] >= 60:
            assessment['compliance_level'] = 'good'
        elif assessment['privacy_score'] >= 40:
            assessment['compliance_level'] = 'moderate'
        else:
            assessment['compliance_level'] = 'needs_improvement'
        
        # Generate recommendations
        recommendations = []
        
        gpc_data = signal_effectiveness.get('gpc')
        if gpc_data and gpc_data['effectiveness_score'] < 60:
            recommendations.append("Improve Global Privacy Control compliance")
        
        cmp_data = signal_effectiveness.get('cmp_reject')
        if cmp_data and cmp_data['effectiveness_score'] < 60:
            recommendations.append("Enhance consent management effectiveness")
        
        baseline_report = scenario_reports.get('baseline')
        if baseline_report:
            non_essential_count = len([c for c in baseline_report.cookies if c.essential is False])
            if non_essential_count > 10:
                recommendations.append("Consider reducing non-essential cookies")
        
        assessment['recommendations'] = recommendations
        
        return assessment
    
    def generate_comparison_report(
        self,
        scenario_diffs: Dict[str, CookieDiff],
        scenario_reports: Dict[str, ScenarioCookieReport]
    ) -> Dict[str, Any]:
        """Generate comprehensive comparison report.
        
        Args:
            scenario_diffs: Scenario differences
            scenario_reports: Scenario reports
            
        Returns:
            Comprehensive comparison analysis
        """
        # Privacy signal effectiveness
        effectiveness_analysis = self.analyze_privacy_signal_effectiveness(scenario_reports)
        
        # Detailed comparison metrics
        comparison_metrics = {}
        
        for scenario_id, diff in scenario_diffs.items():
            comparison_metrics[scenario_id] = {
                'cookie_reduction': diff.reduction_percentage,
                'cookies_removed': len(diff.removed_cookies),
                'cookies_added': len(diff.added_cookies),
                'cookies_modified': len(diff.modified_cookies),
                'policy_improvements': diff.violations_resolved - diff.violations_introduced,
                'significant_change': diff.significant_reduction,
            }
        
        # Summary statistics
        total_scenarios = len(scenario_diffs)
        effective_scenarios = sum(
            1 for diff in scenario_diffs.values() 
            if diff.reduction_percentage > 10
        )
        
        return {
            'summary': {
                'total_comparisons': total_scenarios,
                'effective_scenarios': effective_scenarios,
                'effectiveness_rate': (effective_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0,
                'average_reduction': sum(d.reduction_percentage for d in scenario_diffs.values()) / total_scenarios if total_scenarios > 0 else 0,
            },
            'scenario_metrics': comparison_metrics,
            'privacy_effectiveness': effectiveness_analysis,
            'detailed_diffs': {
                scenario_id: {
                    'baseline_scenario': diff.baseline_scenario,
                    'variant_scenario': diff.variant_scenario,
                    'reduction_percentage': diff.reduction_percentage,
                    'total_changes': diff.total_changes,
                    'policy_improvement': diff.policy_improvement,
                }
                for scenario_id, diff in scenario_diffs.items()
            }
        }

    def compare_cookies(
        self,
        baseline_cookies: List[CookieRecord],
        variant_cookies: List[CookieRecord],
        baseline_scenario_id: str,
        variant_scenario_id: str,
        page_url: str = "test://example.com"
    ) -> 'CookieDiff':
        """Compare two lists of cookies directly (backward compatibility method).

        Args:
            baseline_cookies: Baseline scenario cookies
            variant_cookies: Variant scenario cookies
            baseline_scenario_id: Baseline scenario identifier
            variant_scenario_id: Variant scenario identifier
            page_url: Page URL for the comparison

        Returns:
            Cookie difference analysis with scenario_a_id and scenario_b_id aliases
        """
        from .models import CookieDiff, CookieChange, CookieChangeType

        diff = CookieDiff(
            baseline_scenario=baseline_scenario_id,
            variant_scenario=variant_scenario_id,
            page_url=page_url
        )

        # Create dictionaries for efficient lookup
        baseline_dict = {self._generate_cookie_key(c): c for c in baseline_cookies}
        variant_dict = {self._generate_cookie_key(c): c for c in variant_cookies}

        # Find added cookies (in variant but not baseline)
        for key, cookie in variant_dict.items():
            if key not in baseline_dict:
                diff.added_cookies.append(cookie)

        # Find removed cookies (in baseline but not variant)
        for key, cookie in baseline_dict.items():
            if key not in variant_dict:
                diff.removed_cookies.append(cookie)

        # Find modified and unchanged cookies
        for key in baseline_dict:
            if key in variant_dict:
                baseline_cookie = baseline_dict[key]
                variant_cookie = variant_dict[key]

                # Check for attribute changes
                changes = self._detect_cookie_changes(baseline_cookie, variant_cookie)
                if changes:
                    cookie_change = CookieChange(
                        cookie_key=key,
                        change_type=CookieChangeType.MODIFIED,
                        cookie_name=baseline_cookie.name,
                        cookie_domain=baseline_cookie.domain,
                        cookie_path=baseline_cookie.path,
                        attribute_changes=changes,
                        baseline_value=self._cookie_to_dict(baseline_cookie),
                        variant_value=self._cookie_to_dict(variant_cookie)
                    )
                    diff.modified_cookies.append(cookie_change)
                else:
                    diff.unchanged_cookies.append(baseline_cookie)

        # Calculate reduction percentage
        baseline_count = len(baseline_cookies)
        variant_count = len(variant_cookies)

        if baseline_count > 0:
            diff.reduction_percentage = ((baseline_count - variant_count) / baseline_count) * 100

        # Add backward compatibility aliases
        diff.scenario_a_id = baseline_scenario_id
        diff.scenario_b_id = variant_scenario_id

        return diff

    def analyze_privacy_impact(self, scenario_reports: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze privacy impact across scenarios (stub implementation)."""
        return {
            "privacy_score_improvement": 0.0,
            "cookie_reduction_percentage": 0.0,
            "tracking_blocked": False,
            "compliance_improvement": False,
            "recommendations": []
        }

    def generate_comparison_statistics(self, comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistical analysis of scenario comparisons (stub implementation)."""
        return {
            "total_comparisons": len(comparisons),
            "average_cookie_reduction": 0.0,
            "most_effective_scenario": "unknown",
            "summary": "Statistics not yet implemented"
        }

    def generate_visualization_data(self, diff: 'CookieDiff') -> Dict[str, Any]:
        """Generate data for visualization (stub implementation)."""
        return {
            "chart_type": "cookie_diff",
            "baseline_count": len(diff.removed_cookies) + len(diff.unchanged_cookies),
            "variant_count": len(diff.added_cookies) + len(diff.unchanged_cookies),
            "changes": {
                "added": len(diff.added_cookies),
                "removed": len(diff.removed_cookies),
                "modified": len(diff.modified_cookies)
            }
        }

    def _cookies_match(self, cookie1: CookieRecord, cookie2: CookieRecord) -> bool:
        """Check if two cookies are considered the same (stub implementation)."""
        return (cookie1.name == cookie2.name and
                cookie1.domain == cookie2.domain and
                cookie1.path == cookie2.path)