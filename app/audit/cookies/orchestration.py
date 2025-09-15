"""Scenario orchestration engine for privacy testing workflows.

This module provides comprehensive orchestration of privacy testing scenarios,
coordinating browser contexts, privacy signals, CMP interactions, and cookie
collection across multiple test scenarios with proper isolation.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple
from contextlib import asynccontextmanager

from playwright.async_api import BrowserContext, Page, Browser

from .models import Scenario, CookieRecord, ScenarioCookieReport, ConsentState, PrivacyAnalysisResult
from .config import PrivacyConfiguration, get_privacy_config
from .collection import EnhancedCookieCollector
from .classification import CookieClassifier
from .policy import PolicyComplianceEngine, ComplianceFramework
from .gpc import GPCSimulator
from .cmp import ConsentAutomator, CMPInteractionResult

logger = logging.getLogger(__name__)


class ScenarioExecutionResult:
    """Result of executing a single privacy scenario."""
    
    def __init__(self, scenario_id: str):
        self.scenario_id = scenario_id
        self.success = False
        self.cookies: List[CookieRecord] = []
        self.cmp_result: Optional[CMPInteractionResult] = None
        self.gpc_analysis: Optional[Dict[str, Any]] = None
        self.policy_issues = []
        self.errors = []
        self.execution_time_ms = 0
        self.screenshots = []
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
    
    def mark_complete(self):
        """Mark scenario execution as complete."""
        self.end_time = datetime.utcnow()
        self.execution_time_ms = (self.end_time - self.start_time).total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'scenario_id': self.scenario_id,
            'success': self.success,
            'cookies_count': len(self.cookies),
            'policy_issues_count': len(self.policy_issues),
            'errors': self.errors,
            'execution_time_ms': self.execution_time_ms,
            'screenshots': self.screenshots,
            'cmp_result': self.cmp_result.to_dict() if self.cmp_result else None,
            'gpc_analysis': self.gpc_analysis,
        }


class ScenarioOrchestrator:
    """Orchestrates privacy testing scenarios with proper isolation and coordination.
    
    Manages browser contexts, coordinates privacy signal injection, handles CMP
    interactions, and collects cookies across multiple isolated scenarios.
    """
    
    def __init__(
        self, 
        browser: Browser,
        config: Optional[PrivacyConfiguration] = None,
        artifacts_dir: Optional[Path] = None
    ):
        """Initialize scenario orchestrator.
        
        Args:
            browser: Playwright browser instance
            config: Privacy configuration
            artifacts_dir: Directory for screenshots and artifacts
        """
        self.browser = browser
        self.config = config or get_privacy_config()
        self.artifacts_dir = artifacts_dir or Path("artifacts")
        
        # Components
        self.classifier = CookieClassifier(config)
        self.policy_engine = PolicyComplianceEngine(config)
        self.gpc_simulator = GPCSimulator(config.gpc if config else None)
        self.consent_automator = ConsentAutomator(config.cmp if config else None)
        
        # Execution state
        self.execution_results: Dict[str, ScenarioExecutionResult] = {}
        
        # Ensure artifacts directory exists
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    @asynccontextmanager
    async def create_isolated_context(
        self, 
        scenario: Scenario
    ) -> AsyncGenerator[Tuple[BrowserContext, Page], None]:
        """Create isolated browser context for scenario execution.
        
        Args:
            scenario: Scenario to create context for
            
        Yields:
            Tuple of (context, page) for scenario execution
        """
        context = None
        page = None
        
        try:
            # Create fresh context with scenario-specific settings
            context_options = {
                'ignore_https_errors': True,
                'accept_downloads': False,
            }
            
            # Add extra HTTP headers if scenario specifies them
            if scenario.request_headers:
                context_options['extra_http_headers'] = scenario.request_headers
            
            context = await self.browser.new_context(**context_options)
            
            # Enable GPC simulation if needed
            if scenario.request_headers.get('Sec-GPC'):
                await self.gpc_simulator.enable_gpc_for_context(context)
            
            # Create page
            page = await context.new_page()
            
            logger.info(f"Created isolated context for scenario: {scenario.id}")
            
            yield context, page
            
        except Exception as e:
            logger.error(f"Error in scenario context for {scenario.id}: {e}")
            raise
        finally:
            # Clean up
            try:
                if page:
                    await page.close()
                if context:
                    await context.close()
                logger.debug(f"Cleaned up context for scenario: {scenario.id}")
            except Exception as e:
                logger.warning(f"Error cleaning up context: {e}")
    
    async def execute_scenario(
        self, 
        scenario: Scenario, 
        page_url: str,
        compliance_framework: ComplianceFramework = ComplianceFramework.GDPR
    ) -> ScenarioExecutionResult:
        """Execute a single privacy scenario.
        
        Args:
            scenario: Scenario to execute
            page_url: URL to test
            compliance_framework: Privacy framework for policy validation
            
        Returns:
            Scenario execution result
        """
        result = ScenarioExecutionResult(scenario.id)
        
        try:
            logger.info(f"Executing scenario: {scenario.id} for {page_url}")
            
            async with self.create_isolated_context(scenario) as (context, page):
                # Create cookie collector for this scenario
                collector = EnhancedCookieCollector(context, self.config)
                
                # Navigate to page
                await page.goto(page_url, wait_until='networkidle', timeout=30000)
                result.screenshots.append(await self._take_screenshot(page, f"{scenario.id}_initial"))
                
                # Execute scenario-specific steps (CMP interactions)
                if scenario.steps:
                    cmp_result = await self._execute_cmp_steps(page, scenario)
                    result.cmp_result = cmp_result
                    
                    if cmp_result.success:
                        result.screenshots.extend(cmp_result.screenshots)
                    else:
                        result.errors.extend(cmp_result.errors)
                
                # Wait for network to settle after interactions
                try:
                    await page.wait_for_load_state('networkidle', timeout=5000)
                except Exception as e:
                    logger.debug(f"Network idle wait timeout: {e}")
                
                # Collect cookies
                cookies = await collector.collect_cookies(page_url, scenario.id)
                
                # Classify cookies
                classified_cookies = self.classifier.classify_cookies(cookies, page_url)
                result.cookies = classified_cookies
                
                # Check policy compliance
                policy_issues = self.policy_engine.validate_cookie_policy(
                    classified_cookies, 
                    page_url, 
                    compliance_framework,
                    scenario.id,
                    self.config.environment
                )
                result.policy_issues = policy_issues
                
                # GPC-specific analysis
                if scenario.request_headers.get('Sec-GPC'):
                    gpc_analysis = await self.gpc_simulator.analyze_gpc_response(page, page_url)
                    result.gpc_analysis = gpc_analysis.to_dict()
                
                result.success = True
                
        except Exception as e:
            logger.error(f"Error executing scenario {scenario.id}: {e}")
            result.errors.append(str(e))
        
        finally:
            result.mark_complete()
            self.execution_results[scenario.id] = result
        
        logger.info(f"Scenario {scenario.id} completed in {result.execution_time_ms:.0f}ms")
        return result
    
    async def _execute_cmp_steps(self, page: Page, scenario: Scenario) -> CMPInteractionResult:
        """Execute CMP interaction steps for a scenario.
        
        Args:
            page: Page object
            scenario: Scenario with CMP steps
            
        Returns:
            CMP interaction result
        """
        # Determine consent state from scenario
        consent_state = ConsentState.UNKNOWN
        
        if 'accept_all' in scenario.id:
            consent_state = ConsentState.ACCEPT_ALL
        elif 'reject_all' in scenario.id:
            consent_state = ConsentState.REJECT_ALL
        
        # Generate screenshot path
        screenshot_path = self.artifacts_dir / f"{scenario.id}_cmp_interaction"
        
        return await self.consent_automator.execute_consent_scenario(
            page, consent_state, str(screenshot_path)
        )
    
    async def _take_screenshot(self, page: Page, name_suffix: str) -> str:
        """Take screenshot for debugging/artifacts.
        
        Args:
            page: Page object
            name_suffix: Suffix for screenshot filename
            
        Returns:
            Screenshot file path
        """
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            screenshot_path = self.artifacts_dir / f"screenshot_{name_suffix}_{timestamp}.png"
            await page.screenshot(path=str(screenshot_path))
            return str(screenshot_path)
        except Exception as e:
            logger.warning(f"Failed to take screenshot: {e}")
            return ""
    
    async def execute_all_scenarios(
        self, 
        page_url: str,
        compliance_framework: ComplianceFramework = ComplianceFramework.GDPR,
        parallel_execution: bool = False
    ) -> Dict[str, ScenarioExecutionResult]:
        """Execute all enabled scenarios for a URL.
        
        Args:
            page_url: URL to test
            compliance_framework: Privacy framework for validation
            parallel_execution: Whether to run scenarios in parallel
            
        Returns:
            Dictionary mapping scenario IDs to results
        """
        enabled_scenarios = self.config.get_enabled_scenarios()
        
        if not enabled_scenarios:
            logger.warning("No enabled scenarios found in configuration")
            return {}
        
        logger.info(f"Executing {len(enabled_scenarios)} scenarios for {page_url}")
        
        if parallel_execution:
            # Execute scenarios in parallel
            tasks = [
                self.execute_scenario(scenario, page_url, compliance_framework)
                for scenario in enabled_scenarios
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Scenario {enabled_scenarios[i].id} failed: {result}")
                    error_result = ScenarioExecutionResult(enabled_scenarios[i].id)
                    error_result.errors.append(str(result))
                    error_result.mark_complete()
                    self.execution_results[enabled_scenarios[i].id] = error_result
        else:
            # Execute scenarios sequentially
            for scenario in enabled_scenarios:
                await self.execute_scenario(scenario, page_url, compliance_framework)
        
        return self.execution_results.copy()
    
    async def generate_scenario_reports(
        self, 
        page_url: str,
        page_title: Optional[str] = None
    ) -> Dict[str, ScenarioCookieReport]:
        """Generate detailed reports for each executed scenario.
        
        Args:
            page_url: URL that was tested
            page_title: Optional page title
            
        Returns:
            Dictionary mapping scenario IDs to detailed reports
        """
        reports = {}
        
        for scenario_id, execution_result in self.execution_results.items():
            scenario = self.config.get_scenario_by_id(scenario_id)
            
            if not scenario:
                logger.warning(f"Scenario {scenario_id} not found in config")
                continue
            
            # Create scenario report
            report = ScenarioCookieReport(
                scenario_id=scenario_id,
                scenario_name=scenario.name,
                page_url=page_url,
                page_title=page_title,
                cookies=execution_result.cookies,
                policy_issues=execution_result.policy_issues,
                errors=execution_result.errors
            )
            
            # Set consent state if CMP was used
            if execution_result.cmp_result:
                report.consent_state = execution_result.cmp_result.consent_state
            
            reports[scenario_id] = report
        
        return reports
    
    def get_baseline_cookies(self) -> Optional[List[CookieRecord]]:
        """Get cookies from baseline scenario for comparison.
        
        Returns:
            Baseline scenario cookies, or None if not available
        """
        baseline_result = self.execution_results.get('baseline')
        return baseline_result.cookies if baseline_result else None
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all scenario executions.
        
        Returns:
            Execution summary statistics
        """
        total_scenarios = len(self.execution_results)
        successful_scenarios = sum(1 for r in self.execution_results.values() if r.success)
        total_cookies_collected = sum(len(r.cookies) for r in self.execution_results.values())
        total_policy_issues = sum(len(r.policy_issues) for r in self.execution_results.values())
        total_execution_time = sum(r.execution_time_ms for r in self.execution_results.values())
        
        return {
            'total_scenarios': total_scenarios,
            'successful_scenarios': successful_scenarios,
            'success_rate': (successful_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0,
            'total_cookies_collected': total_cookies_collected,
            'total_policy_issues': total_policy_issues,
            'total_execution_time_ms': total_execution_time,
            'average_execution_time_ms': total_execution_time / total_scenarios if total_scenarios > 0 else 0,
            'scenarios_with_errors': [r.scenario_id for r in self.execution_results.values() if r.errors],
            'scenarios_with_cmp_interactions': [
                r.scenario_id for r in self.execution_results.values() 
                if r.cmp_result and r.cmp_result.success
            ],
        }
    
    async def cleanup(self):
        """Clean up orchestrator resources."""
        try:
            # Disable GPC for any remaining contexts
            for context_id in list(self.gpc_simulator.injected_contexts):
                # Context should already be closed, but clean up tracking
                self.gpc_simulator.injected_contexts.discard(context_id)
            
            logger.info("Scenario orchestrator cleaned up")
        except Exception as e:
            logger.warning(f"Error during orchestrator cleanup: {e}")
    
    def __repr__(self) -> str:
        """String representation of orchestrator."""
        return (f"ScenarioOrchestrator("
                f"scenarios={len(self.config.scenarios)}, "
                f"executed={len(self.execution_results)})")


async def execute_privacy_scenarios(
    browser: Browser,
    page_url: str,
    config: Optional[PrivacyConfiguration] = None,
    artifacts_dir: Optional[Path] = None,
    parallel_execution: bool = False,
    compliance_framework: ComplianceFramework = ComplianceFramework.GDPR
) -> PrivacyAnalysisResult:
    """Convenience function to execute complete privacy scenario analysis.
    
    Args:
        browser: Playwright browser instance
        page_url: URL to analyze
        config: Privacy configuration
        artifacts_dir: Directory for artifacts
        parallel_execution: Whether to run scenarios in parallel
        compliance_framework: Privacy framework for compliance validation
        
    Returns:
        Complete privacy analysis result
    """
    orchestrator = ScenarioOrchestrator(browser, config, artifacts_dir)
    
    try:
        # Execute all scenarios
        execution_results = await orchestrator.execute_all_scenarios(
            page_url, compliance_framework, parallel_execution
        )
        
        # Generate detailed reports
        scenario_reports = await orchestrator.generate_scenario_reports(page_url)
        
        # Create comprehensive analysis result
        analysis = PrivacyAnalysisResult(
            page_url=page_url,
            scenario_reports=scenario_reports,
            config=config  # Include the effective PrivacyConfiguration
        )
        
        # Add execution metadata
        execution_summary = orchestrator.get_execution_summary()
        logger.info(f"Privacy analysis complete: {execution_summary['success_rate']:.1f}% success rate")
        
        return analysis
        
    finally:
        await orchestrator.cleanup()