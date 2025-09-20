"""Main Cookie & Consent Service for Epic 5.

This module provides the primary interface for cookie and consent management,
orchestrating all privacy testing workflows and providing a clean API for
integration with the main audit runner.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from playwright.async_api import Browser, BrowserContext

from .models import PrivacyAnalysisResult, ScenarioCookieReport, CookieDiff, PolicySeverity
from .config import PrivacyConfiguration, get_privacy_config
from .orchestration import ScenarioOrchestrator, execute_privacy_scenarios
from .comparison import ScenarioComparator
from .policy import PolicyComplianceEngine, ComplianceFramework

logger = logging.getLogger(__name__)


class CookieConsentService:
    """Main service orchestrating all cookie and consent operations.
    
    Provides the primary interface for Epic 5 functionality, coordinating
    privacy testing workflows, scenario execution, and results analysis.
    """
    
    def __init__(
        self,
        browser: Optional[Browser] = None,
        config: Optional[PrivacyConfiguration] = None,
        artifacts_dir: Optional[Path] = None
    ):
        """Initialize Cookie & Consent Service.
        
        Args:
            browser: Playwright browser instance
            config: Privacy configuration
            artifacts_dir: Directory for artifacts and screenshots
        """
        self.browser = browser
        self.config = config or get_privacy_config()
        self.artifacts_dir = artifacts_dir or Path("artifacts/privacy")
        
        # Components
        self.comparator = ScenarioComparator()
        self.policy_engine = PolicyComplianceEngine(self.config)
        
        # Service state
        self.last_analysis: Optional[PrivacyAnalysisResult] = None
        self.service_metrics = {
            'analyses_performed': 0,
            'scenarios_executed': 0,
            'cookies_analyzed': 0,
            'policy_issues_found': 0,
        }
        
        # Ensure artifacts directory exists
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    async def analyze_page_privacy(
        self,
        page_url: str,
        page_title: Optional[str] = None,  # Added for backward compatibility
        compliance_framework: ComplianceFramework = ComplianceFramework.GDPR,
        parallel_execution: bool = False,
        include_performance_metrics: bool = True
    ) -> PrivacyAnalysisResult:
        """Perform comprehensive privacy analysis of a web page.

        Args:
            page_url: URL to analyze
            page_title: Optional page title (for backward compatibility)
            compliance_framework: Privacy framework for compliance validation
            parallel_execution: Whether to run scenarios in parallel
            include_performance_metrics: Whether to include detailed performance metrics

        Returns:
            Comprehensive privacy analysis result
        """
        if not self.browser:
            raise RuntimeError("Browser instance required for privacy analysis")
        
        start_time = datetime.utcnow()
        logger.info(f"Starting privacy analysis for: {page_url}")
        
        try:
            # Execute privacy scenarios
            analysis = await execute_privacy_scenarios(
                self.browser,
                page_url,
                self.config,
                self.artifacts_dir,
                parallel_execution,
                compliance_framework
            )
            
            # Perform cross-scenario analysis
            scenario_diffs = await self._perform_cross_scenario_analysis(analysis.scenario_reports)
            analysis.scenario_diffs = scenario_diffs
            
            # Calculate privacy scores
            privacy_metrics = await self._calculate_privacy_metrics(analysis)
            analysis.privacy_score = privacy_metrics['overall_score']
            analysis.gpc_effectiveness = privacy_metrics.get('gpc_effectiveness')
            analysis.cmp_effectiveness = privacy_metrics.get('cmp_effectiveness')
            
            # Generate recommendations
            analysis.recommendations = self._generate_privacy_recommendations(analysis)
            
            # Identify critical issues
            analysis.critical_issues = self._identify_critical_issues(analysis)
            
            # Update service metrics
            self._update_service_metrics(analysis)
            
            # Store for future reference
            self.last_analysis = analysis
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Privacy analysis completed in {execution_time:.2f}s")
            
            if include_performance_metrics:
                analysis.metadata = analysis.metadata or {}
                analysis.metadata['execution_time_seconds'] = execution_time
                analysis.metadata['service_metrics'] = self.service_metrics.copy()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Privacy analysis failed for {page_url}: {e}")
            raise
    
    async def _perform_cross_scenario_analysis(
        self, 
        scenario_reports: Dict[str, ScenarioCookieReport]
    ) -> Dict[str, CookieDiff]:
        """Perform cross-scenario differential analysis.
        
        Args:
            scenario_reports: Dictionary of scenario reports
            
        Returns:
            Dictionary mapping scenario pairs to their differences
        """
        if 'baseline' not in scenario_reports:
            logger.warning("No baseline scenario found for comparison")
            return {}
        
        # Compare all scenarios against baseline
        diffs = self.comparator.compare_multiple_scenarios(scenario_reports, 'baseline')
        
        # Log comparison summary
        for scenario_id, diff in diffs.items():
            logger.info(
                f"Scenario comparison {scenario_id}: "
                f"{diff.reduction_percentage:.1f}% cookie reduction, "
                f"{diff.total_changes} total changes"
            )
        
        return diffs
    
    async def _calculate_privacy_metrics(self, analysis: PrivacyAnalysisResult) -> Dict[str, float]:
        """Calculate privacy effectiveness metrics.
        
        Args:
            analysis: Privacy analysis result
            
        Returns:
            Dictionary of privacy metrics
        """
        metrics = {}
        
        baseline_report = analysis.baseline_report
        if not baseline_report:
            logger.warning("No baseline report available for metrics calculation")
            return {'overall_score': 0.0}
        
        baseline_cookie_count = len(baseline_report.cookies)
        baseline_issues_count = len(baseline_report.policy_issues)
        
        # Calculate GPC effectiveness
        gpc_report = analysis.gpc_report
        if gpc_report and analysis.scenario_diffs.get('gpc_on'):
            gpc_diff = analysis.scenario_diffs['gpc_on']
            if baseline_cookie_count > 0:
                metrics['gpc_effectiveness'] = gpc_diff.reduction_percentage
            else:
                metrics['gpc_effectiveness'] = 100.0  # No cookies to reduce
        
        # Calculate CMP effectiveness
        cmp_reject_report = analysis.scenario_reports.get('cmp_reject_all')
        if cmp_reject_report and analysis.scenario_diffs.get('cmp_reject_all'):
            cmp_diff = analysis.scenario_diffs['cmp_reject_all']
            if baseline_cookie_count > 0:
                metrics['cmp_effectiveness'] = cmp_diff.reduction_percentage
        
        # Calculate overall privacy score
        score_components = []
        
        # Cookie privacy score (40%)
        if baseline_cookie_count > 0:
            third_party_ratio = len([c for c in baseline_report.cookies if not c.is_first_party]) / baseline_cookie_count
            analytics_ratio = len([
                c for c in baseline_report.cookies 
                if c.metadata and c.metadata.get('classification', {}).get('category') == 'analytics'
            ]) / baseline_cookie_count
            
            cookie_score = max(0, 100 - (third_party_ratio * 30 + analytics_ratio * 30))
            score_components.append(('cookies', cookie_score, 0.4))
        
        # Policy compliance score (30%)
        if baseline_cookie_count > 0:
            compliance_score = max(0, 100 - (baseline_issues_count / baseline_cookie_count * 20))
            score_components.append(('compliance', compliance_score, 0.3))
        
        # Privacy signal effectiveness (30%)
        signal_scores = []
        if 'gpc_effectiveness' in metrics:
            signal_scores.append(min(100, metrics['gpc_effectiveness'] * 2))  # Amplify effectiveness
        if 'cmp_effectiveness' in metrics:
            signal_scores.append(min(100, metrics['cmp_effectiveness'] * 2))
        
        if signal_scores:
            avg_signal_score = sum(signal_scores) / len(signal_scores)
            score_components.append(('signals', avg_signal_score, 0.3))
        
        # Calculate weighted average
        if score_components:
            total_weight = sum(weight for _, _, weight in score_components)
            weighted_sum = sum(score * weight for _, score, weight in score_components)
            metrics['overall_score'] = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            metrics['overall_score'] = 0.0
        
        return metrics
    
    def _generate_privacy_recommendations(self, analysis: PrivacyAnalysisResult) -> List[str]:
        """Generate privacy improvement recommendations.
        
        Args:
            analysis: Privacy analysis result
            
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        
        baseline_report = analysis.baseline_report
        if not baseline_report:
            return ["Unable to generate recommendations without baseline analysis"]
        
        baseline_cookies = baseline_report.cookies
        total_cookies = len(baseline_cookies)
        
        if total_cookies == 0:
            return ["No cookies found - excellent privacy posture"]
        
        # Third-party cookie recommendations
        third_party_count = len([c for c in baseline_cookies if not c.is_first_party])
        if third_party_count > total_cookies * 0.3:  # More than 30% third-party
            recommendations.append(
                f"Reduce third-party cookies: {third_party_count}/{total_cookies} "
                f"({third_party_count/total_cookies*100:.1f}%) are third-party"
            )
        
        # Analytics cookie recommendations
        analytics_cookies = [
            c for c in baseline_cookies 
            if c.metadata and c.metadata.get('classification', {}).get('category') == 'analytics'
        ]
        if analytics_cookies:
            recommendations.append(
                f"Implement consent management for {len(analytics_cookies)} analytics cookies"
            )
        
        # GPC effectiveness recommendations
        if analysis.gpc_effectiveness is not None and analysis.gpc_effectiveness < 30:
            recommendations.append("Improve Global Privacy Control compliance - current effectiveness is low")
        
        # CMP effectiveness recommendations
        if analysis.cmp_effectiveness is not None and analysis.cmp_effectiveness < 50:
            recommendations.append("Enhance consent management effectiveness - many cookies persist after rejection")
        
        # Security attribute recommendations
        security_issues = [
            issue for report in analysis.scenario_reports.values()
            for issue in report.policy_issues
            if issue.attribute in ['secure', 'http_only', 'same_site']
        ]
        if security_issues:
            recommendations.append(f"Fix {len(security_issues)} cookie security issues")
        
        # Privacy score recommendations
        if analysis.privacy_score < 60:
            recommendations.insert(0, "Overall privacy score is below recommended threshold - prioritize improvements")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _identify_critical_issues(self, analysis: PrivacyAnalysisResult) -> List[Any]:
        """Identify critical privacy issues across all scenarios.
        
        Args:
            analysis: Privacy analysis result
            
        Returns:
            List of critical issues
        """
        critical_issues = []
        
        for scenario_id, report in analysis.scenario_reports.items():
            # Find critical policy violations
            critical_violations = [
                issue for issue in report.policy_issues
                if issue.severity == PolicySeverity.CRITICAL
            ]
            
            for violation in critical_violations:
                critical_issues.append({
                    'type': 'policy_violation',
                    'scenario': scenario_id,
                    'issue': violation,
                    'severity': 'critical'
                })
        
        # Check for privacy signal non-compliance
        if analysis.scenario_diffs.get('gpc_on'):
            gpc_diff = analysis.scenario_diffs['gpc_on']
            if gpc_diff.reduction_percentage < 10:  # Less than 10% reduction
                critical_issues.append({
                    'type': 'gpc_non_compliance',
                    'message': 'Site does not adequately respect Global Privacy Control signals',
                    'severity': 'high'
                })
        
        return critical_issues
    
    def _update_service_metrics(self, analysis: PrivacyAnalysisResult) -> None:
        """Update service-level metrics.
        
        Args:
            analysis: Completed privacy analysis
        """
        self.service_metrics['analyses_performed'] += 1
        self.service_metrics['scenarios_executed'] += len(analysis.scenario_reports)
        self.service_metrics['cookies_analyzed'] += sum(
            len(report.cookies) for report in analysis.scenario_reports.values()
        )
        self.service_metrics['policy_issues_found'] += sum(
            len(report.policy_issues) for report in analysis.scenario_reports.values()
        )
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status and metrics.
        
        Returns:
            Service status information
        """
        return {
            'service': 'CookieConsentService',
            'status': 'active',
            'browser_available': self.browser is not None,
            'config_loaded': self.config is not None,
            'scenarios_configured': len(self.config.scenarios) if self.config else 0,
            'enabled_scenarios': len(self.config.get_enabled_scenarios()) if self.config else 0,
            'artifacts_directory': str(self.artifacts_dir),
            'metrics': self.service_metrics.copy(),
            'last_analysis_time': self.last_analysis.analysis_time.isoformat() if self.last_analysis else None,
        }
    
    def export_analysis_results(
        self, 
        analysis: Optional[PrivacyAnalysisResult] = None,
        export_format: str = 'json'
    ) -> Dict[str, Any]:
        """Export privacy analysis results.
        
        Args:
            analysis: Analysis to export (uses last analysis if not provided)
            export_format: Export format ('json', 'summary')
            
        Returns:
            Exported analysis data
        """
        target_analysis = analysis or self.last_analysis
        if not target_analysis:
            raise ValueError("No analysis available for export")
        
        if export_format == 'summary':
            return self._export_analysis_summary(target_analysis)
        else:  # json format
            return self._export_analysis_full(target_analysis)
    
    def _export_analysis_summary(self, analysis: PrivacyAnalysisResult) -> Dict[str, Any]:
        """Export summary of privacy analysis.
        
        Args:
            analysis: Analysis to summarize
            
        Returns:
            Summary data
        """
        baseline_report = analysis.baseline_report
        
        return {
            'page_url': analysis.page_url,
            'analysis_time': analysis.analysis_time.isoformat(),
            'privacy_score': analysis.privacy_score,
            'total_scenarios': len(analysis.scenario_reports),
            'baseline_cookies': len(baseline_report.cookies) if baseline_report else 0,
            'gpc_effectiveness': analysis.gpc_effectiveness,
            'cmp_effectiveness': analysis.cmp_effectiveness,
            'critical_issues_count': len(analysis.critical_issues),
            'recommendations_count': len(analysis.recommendations),
            'scenario_summary': {
                scenario_id: {
                    'cookies': len(report.cookies),
                    'policy_issues': len(report.policy_issues),
                    'consent_state': report.consent_state.value if report.consent_state else None
                }
                for scenario_id, report in analysis.scenario_reports.items()
            }
        }
    
    def _export_analysis_full(self, analysis: PrivacyAnalysisResult) -> Dict[str, Any]:
        """Export full privacy analysis data.
        
        Args:
            analysis: Analysis to export
            
        Returns:
            Complete analysis data
        """
        # Convert analysis to dictionary format
        # This would normally use analysis.model_dump() but we'll create manually
        return {
            'page_url': analysis.page_url,
            'page_title': analysis.page_title,
            'analysis_time': analysis.analysis_time.isoformat(),
            'privacy_score': analysis.privacy_score,
            'gpc_effectiveness': analysis.gpc_effectiveness,
            'cmp_effectiveness': analysis.cmp_effectiveness,
            'scenario_reports': {
                scenario_id: {
                    'scenario_id': report.scenario_id,
                    'scenario_name': report.scenario_name,
                    'total_cookies': report.total_cookies,
                    'first_party_cookies': report.first_party_cookies,
                    'third_party_cookies': report.third_party_cookies,
                    'essential_cookies': report.essential_cookies,
                    'policy_issues_count': len(report.policy_issues),
                    'consent_state': report.consent_state.value if report.consent_state else None,
                    'has_violations': report.has_violations,
                }
                for scenario_id, report in analysis.scenario_reports.items()
            },
            'scenario_diffs': {
                scenario_id: {
                    'reduction_percentage': diff.reduction_percentage,
                    'cookies_removed': len(diff.removed_cookies),
                    'cookies_added': len(diff.added_cookies),
                    'total_changes': diff.total_changes,
                    'policy_improvement': diff.policy_improvement,
                }
                for scenario_id, diff in analysis.scenario_diffs.items()
            },
            'critical_issues': analysis.critical_issues,
            'recommendations': analysis.recommendations,
        }
    
    async def analyze_multiple_pages(
        self,
        page_urls: List[str],
        compliance_framework: ComplianceFramework = ComplianceFramework.GDPR,
        parallel_execution: bool = False,
        include_performance_metrics: bool = True
    ) -> List[PrivacyAnalysisResult]:
        """Analyze multiple pages for privacy compliance.
        
        Args:
            page_urls: List of URLs to analyze
            compliance_framework: Privacy framework for compliance validation
            parallel_execution: Whether to run scenarios in parallel for each page
            include_performance_metrics: Whether to include detailed performance metrics
            
        Returns:
            List of privacy analysis results for each page
        """
        if not self.browser:
            raise RuntimeError("Browser instance required for privacy analysis")
        
        logger.info(f"Starting analysis of {len(page_urls)} pages")
        results = []
        
        for page_url in page_urls:
            try:
                result = await self.analyze_page_privacy(
                    page_url,
                    page_title=None,
                    compliance_framework=compliance_framework,
                    parallel_execution=parallel_execution,
                    include_performance_metrics=include_performance_metrics
                )
                results.append(result)
                logger.debug(f"Completed analysis for: {page_url}")
            except Exception as e:
                logger.error(f"Failed to analyze {page_url}: {e}")
                # Continue with other URLs even if one fails
                continue
        
        logger.info(f"Completed analysis of {len(results)}/{len(page_urls)} pages")
        return results
    
    async def batch_privacy_analysis(
        self,
        page_urls: List[str],
        max_concurrent: int = 3,
        compliance_framework: ComplianceFramework = ComplianceFramework.GDPR,
        parallel_execution: bool = False
    ) -> List[PrivacyAnalysisResult]:
        """Run privacy analysis on multiple pages with concurrent processing.
        
        Args:
            page_urls: List of URLs to analyze
            max_concurrent: Maximum number of concurrent page analyses
            compliance_framework: Privacy framework for compliance validation
            parallel_execution: Whether to run scenarios in parallel for each page
            
        Returns:
            List of privacy analysis results for all successfully analyzed pages
        """
        import asyncio
        
        if not self.browser:
            raise RuntimeError("Browser instance required for privacy analysis")
        
        logger.info(f"Starting batch analysis of {len(page_urls)} pages with max_concurrent={max_concurrent}")
        
        # Create semaphore to limit concurrent analyses
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(url: str) -> Optional[PrivacyAnalysisResult]:
            async with semaphore:
                try:
                    return await self.analyze_page_privacy(
                        url,
                        page_title=None,
                        compliance_framework=compliance_framework,
                        parallel_execution=parallel_execution,
                        include_performance_metrics=False  # Skip metrics for batch to improve performance
                    )
                except Exception as e:
                    logger.error(f"Failed to analyze {url} in batch: {e}")
                    return None
        
        # Execute all analyses concurrently
        tasks = [analyze_with_semaphore(url) for url in page_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        successful_results = [
            result for result in results 
            if isinstance(result, PrivacyAnalysisResult)
        ]
        
        logger.info(f"Batch analysis completed: {len(successful_results)}/{len(page_urls)} pages successful")
        return successful_results
    
    async def cleanup(self):
        """Clean up service resources."""
        try:
            # Clear caches
            self.comparator.comparison_cache.clear()
            
            # Reset metrics if needed
            logger.info("Cookie & Consent Service cleaned up")
        except Exception as e:
            logger.warning(f"Error during service cleanup: {e}")
    
    def __repr__(self) -> str:
        """String representation of service."""
        return (f"CookieConsentService("
                f"analyses={self.service_metrics['analyses_performed']}, "
                f"scenarios_configured={len(self.config.scenarios) if self.config else 0})")


# Convenience functions for backward compatibility and easy integration

async def analyze_page_privacy(
    browser: Browser,
    page_url: str,
    page_title: Optional[str] = None,
    config: Optional[PrivacyConfiguration] = None,
    artifacts_dir: Optional[Path] = None,
    compliance_framework: ComplianceFramework = ComplianceFramework.GDPR,
    parallel_execution: bool = False
) -> PrivacyAnalysisResult:
    """Convenience function for single-page privacy analysis.

    Args:
        browser: Playwright browser instance
        page_url: URL to analyze
        page_title: Optional page title for analysis context (preserved for backward compatibility)
        config: Privacy configuration
        artifacts_dir: Directory for artifacts and screenshots
        compliance_framework: Privacy framework for validation
        parallel_execution: Whether to run scenarios in parallel

    Returns:
        Privacy analysis result
    """
    service = CookieConsentService(browser, config, artifacts_dir)
    try:
        return await service.analyze_page_privacy(
            page_url,
            page_title=page_title,
            compliance_framework=compliance_framework,
            parallel_execution=parallel_execution
        )
    finally:
        # Handle cleanup gracefully
        cleanup_method = getattr(service, 'cleanup', None)
        if cleanup_method and callable(cleanup_method):
            try:
                await service.cleanup()
            except TypeError:
                # Mock object - call synchronously
                service.cleanup()


def create_cookie_consent_service(
    browser: Browser,
    config_path_or_config: Optional[Union[Path, 'PrivacyConfiguration']] = None,
    artifacts_dir: Optional[Path] = None
) -> CookieConsentService:
    """Create configured Cookie & Consent Service.

    Args:
        browser: Playwright browser instance
        config_path_or_config: Path to privacy configuration file or PrivacyConfiguration object
        artifacts_dir: Directory for artifacts

    Returns:
        Configured service instance
    """
    from .config import PrivacyConfiguration, load_privacy_config_from_file

    config = None
    if config_path_or_config:
        if isinstance(config_path_or_config, PrivacyConfiguration):
            config = config_path_or_config
        else:
            config = load_privacy_config_from_file(config_path_or_config)

    return CookieConsentService(browser, config, artifacts_dir)

