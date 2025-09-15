"""Unit tests for scenario orchestration engine."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from pathlib import Path

from playwright.async_api import Browser, BrowserContext, Page

from app.audit.cookies.orchestration import ScenarioOrchestrator, ScenarioExecutionResult
from app.audit.cookies.models import Scenario, CookieRecord, ConsentState
from app.audit.cookies.config import PrivacyConfiguration


class TestScenarioExecutionResult:
    """Test ScenarioExecutionResult functionality."""
    
    def test_result_initialization(self):
        """Test result object initialization."""
        result = ScenarioExecutionResult("test_scenario")
        
        assert result.scenario_id == "test_scenario"
        assert result.success is False
        assert result.cookies == []
        assert result.cmp_result is None
        assert result.policy_issues == []
        assert result.errors == []
        assert result.execution_time_ms == 0
        assert result.end_time is None
        assert isinstance(result.start_time, datetime)
    
    def test_mark_complete(self):
        """Test marking result as complete."""
        result = ScenarioExecutionResult("test")
        
        # Mark complete
        result.mark_complete()
        
        assert result.end_time is not None
        assert result.execution_time_ms > 0
        assert result.end_time >= result.start_time
    
    def test_result_serialization(self):
        """Test result to dictionary conversion.""" 
        result = ScenarioExecutionResult("test")
        result.success = True
        result.cookies = [
            CookieRecord(
                name="test_cookie",
                value="value",
                domain="example.com",
                path="/",
                secure=True,
                http_only=False,
                first_party=True,
                scenario_id="test"
            )
        ]
        result.errors = ["Test error"]
        result.mark_complete()
        
        data = result.to_dict()
        
        assert data["scenario_id"] == "test"
        assert data["success"] is True
        assert data["cookies_count"] == 1
        assert data["errors"] == ["Test error"]
        assert data["execution_time_ms"] > 0


class TestScenarioOrchestrator:
    """Test ScenarioOrchestrator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_browser = Mock(spec=Browser)
        self.config = PrivacyConfiguration()
        self.artifacts_dir = Path("/tmp/test_artifacts")
        
        self.orchestrator = ScenarioOrchestrator(
            self.mock_browser, 
            self.config,
            self.artifacts_dir
        )
    
    @pytest.mark.asyncio
    async def test_isolated_context_creation(self):
        """Test creation of isolated browser context."""
        # Mock scenario
        scenario = Scenario(
            id="test_scenario",
            name="Test Scenario",
            description="Test scenario for isolated context creation",
            request_headers={"Sec-GPC": "1"}
        )
        
        # Mock browser context and page
        mock_context = Mock(spec=BrowserContext)
        mock_page = Mock(spec=Page)
        
        self.mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()
        mock_page.close = AsyncMock()
        
        # Test context creation
        async with self.orchestrator.create_isolated_context(scenario) as (context, page):
            assert context == mock_context
            assert page == mock_page
            
            # Verify context was created with correct options
            self.mock_browser.new_context.assert_called_once()
            call_kwargs = self.mock_browser.new_context.call_args[1]
            assert call_kwargs["ignore_https_errors"] is True
            assert call_kwargs["accept_downloads"] is False
            assert call_kwargs["extra_http_headers"] == {"Sec-GPC": "1"}
        
        # Verify cleanup
        mock_page.close.assert_called_once()
        mock_context.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_scenario_execution_success(self):
        """Test successful scenario execution."""
        # Mock scenario
        scenario = Scenario(
            id="baseline",
            name="Baseline Test",
            description="Baseline test scenario for execution success"
        )
        
        # Mock dependencies
        mock_context = Mock(spec=BrowserContext)
        mock_page = Mock(spec=Page)
        
        # Mock context creation
        with patch.object(self.orchestrator, 'create_isolated_context') as mock_create:
            mock_create.return_value.__aenter__ = AsyncMock(return_value=(mock_context, mock_page))
            mock_create.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock page navigation
            mock_page.goto = AsyncMock()
            mock_page.wait_for_load_state = AsyncMock()
            
            # Mock cookie collection
            mock_collector = Mock()
            mock_collector.collect_cookies = AsyncMock(return_value=[
                CookieRecord(
                    name="test_cookie",
                    value="value", 
                    domain="example.com",
                    path="/",
                    secure=True,
                    http_only=False,
                    first_party=True,
                    scenario_id="baseline"
                )
            ])
            
            with patch('app.audit.cookies.orchestration.EnhancedCookieCollector', return_value=mock_collector):
                # Mock classification
                self.orchestrator.classifier.classify_cookies = Mock(return_value=[
                    CookieRecord(
                        name="test_cookie",
                        value="value",
                        domain="example.com", 
                        path="/",
                        secure=True,
                        http_only=False,
                        first_party=True,
                        scenario_id="baseline"
                    )
                ])
                
                # Mock policy validation
                self.orchestrator.policy_engine.validate_cookie_policy = Mock(return_value=[])
                
                # Mock screenshot
                with patch.object(self.orchestrator, '_take_screenshot', return_value="/path/to/screenshot.png"):
                    
                    # Execute scenario
                    result = await self.orchestrator.execute_scenario(scenario, "https://example.com")
        
        # Verify result
        assert result.scenario_id == "baseline"
        assert result.success is True
        assert len(result.cookies) == 1
        assert result.cookies[0].name == "test_cookie"
        assert len(result.policy_issues) == 0
        assert result.execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_scenario_execution_with_cmp(self):
        """Test scenario execution with CMP interactions."""
        # Mock scenario with CMP steps
        scenario = Scenario(
            id="cmp_accept_all",
            name="Accept All Cookies",
            description="Test scenario for CMP accept all cookies interaction",
            steps=[{"action": "click", "selector": "[data-testid='accept-all']"}]
        )
        
        # Mock CMP interaction result
        from app.audit.cookies.cmp import CMPInteractionResult
        mock_cmp_result = CMPInteractionResult()
        mock_cmp_result.success = True
        mock_cmp_result.consent_state = ConsentState.ACCEPT_ALL
        mock_cmp_result.interaction_steps = ["click accept-all"]
        mock_cmp_result.screenshots = ["/path/to/cmp_screenshot.png"]
        mock_cmp_result.errors = []
        
        # Mock dependencies
        mock_context = Mock(spec=BrowserContext)
        mock_page = Mock(spec=Page)
        
        with patch.object(self.orchestrator, 'create_isolated_context') as mock_create:
            mock_create.return_value.__aenter__ = AsyncMock(return_value=(mock_context, mock_page))
            mock_create.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock page operations
            mock_page.goto = AsyncMock()
            mock_page.wait_for_load_state = AsyncMock()
            
            # Mock CMP execution
            with patch.object(self.orchestrator, '_execute_cmp_steps', return_value=mock_cmp_result):
                
                # Mock cookie collection and classification
                with patch('app.audit.cookies.orchestration.EnhancedCookieCollector') as mock_collector_class:
                    mock_collector = Mock()
                    mock_collector.collect_cookies = AsyncMock(return_value=[])
                    mock_collector_class.return_value = mock_collector
                    
                    self.orchestrator.classifier.classify_cookies = Mock(return_value=[])
                    self.orchestrator.policy_engine.validate_cookie_policy = Mock(return_value=[])
                    
                    with patch.object(self.orchestrator, '_take_screenshot', return_value="/screenshot.png"):
                        
                        # Execute scenario
                        result = await self.orchestrator.execute_scenario(scenario, "https://example.com")
        
        # Verify CMP interaction was executed
        assert result.cmp_result is not None
        assert result.cmp_result.success is True
        assert result.cmp_result.consent_state == ConsentState.ACCEPT_ALL
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_scenario_execution_failure(self):
        """Test scenario execution with failures."""
        scenario = Scenario(id="failing_scenario", name="Failing Test", description="Test scenario to verify failure handling")
        
        # Mock context creation to raise exception
        with patch.object(self.orchestrator, 'create_isolated_context') as mock_create:
            mock_create.return_value.__aenter__ = AsyncMock(side_effect=Exception("Navigation failed"))
            mock_create.return_value.__aexit__ = AsyncMock()
            
            # Execute scenario
            result = await self.orchestrator.execute_scenario(scenario, "https://example.com")
        
        # Verify failure handling
        assert result.success is False
        assert len(result.errors) == 1
        assert "Navigation failed" in result.errors[0]
        assert result.execution_time_ms > 0  # Should still track timing
    
    @pytest.mark.asyncio
    async def test_execute_all_scenarios_sequential(self):
        """Test executing all scenarios sequentially."""
        # Mock scenarios in config
        scenarios = [
            Scenario(id="baseline", name="Baseline", description="Baseline scenario without privacy settings"),
            Scenario(id="gpc_on", name="GPC Enabled", description="Scenario with GPC signal enabled", request_headers={"Sec-GPC": "1"})
        ]

        # Set scenarios directly on config (they all have enabled=True by default)
        original_scenarios = self.config.scenarios
        self.config.scenarios = scenarios

        try:
            # Mock scenario execution to properly set execution_results
            async def mock_execute_scenario(scenario, page_url, compliance_framework=None):
                result = ScenarioExecutionResult(scenario.id)
                self.orchestrator.execution_results[scenario.id] = result
                return result

            with patch.object(self.orchestrator, 'execute_scenario', side_effect=mock_execute_scenario):
                # Execute all scenarios
                results = await self.orchestrator.execute_all_scenarios(
                    "https://example.com",
                    parallel_execution=False
                )
        finally:
            # Restore original scenarios
            self.config.scenarios = original_scenarios
        
        # Verify execution
        assert len(results) == 2
        assert "baseline" in results
        assert "gpc_on" in results
    
    @pytest.mark.asyncio 
    async def test_execute_all_scenarios_parallel(self):
        """Test executing all scenarios in parallel."""
        # Mock scenarios
        scenarios = [
            Scenario(id="baseline", name="Baseline", description="Baseline scenario for parallel execution"),
            Scenario(id="gpc_on", name="GPC Enabled", description="GPC enabled scenario for parallel execution")
        ]

        # Set scenarios directly on config
        original_scenarios = self.config.scenarios
        self.config.scenarios = scenarios

        try:
            # Mock scenario execution properly
            async def mock_execute_scenario(scenario, page_url, compliance_framework=None):
                result = ScenarioExecutionResult(scenario.id)
                self.orchestrator.execution_results[scenario.id] = result
                return result

            with patch.object(self.orchestrator, 'execute_scenario', side_effect=mock_execute_scenario):
                # Execute all scenarios in parallel
                results = await self.orchestrator.execute_all_scenarios(
                    "https://example.com",
                    parallel_execution=True
                )
        finally:
            # Restore original scenarios
            self.config.scenarios = original_scenarios
        
        # Verify parallel execution results
        assert len(results) == 2
        assert "baseline" in results
        assert "gpc_on" in results
    
    @pytest.mark.asyncio
    async def test_generate_scenario_reports(self):
        """Test scenario report generation."""
        # Mock execution results
        execution_result = ScenarioExecutionResult("baseline")
        execution_result.success = True
        execution_result.cookies = [
            CookieRecord(
                name="test_cookie",
                value="value",
                domain="example.com",
                path="/",
                secure=True, 
                http_only=False,
                first_party=True,
                scenario_id="baseline"
            )
        ]
        execution_result.policy_issues = []
        execution_result.errors = []
        
        self.orchestrator.execution_results = {"baseline": execution_result}

        # Add mock scenario to config scenarios list
        mock_scenario = Scenario(id="baseline", name="Baseline Test", description="Baseline test scenario for report generation")
        original_scenarios = self.config.scenarios
        self.config.scenarios = [mock_scenario]

        try:
            # Generate reports
            reports = await self.orchestrator.generate_scenario_reports(
                "https://example.com", "Test Page"
            )
        finally:
            # Restore original scenarios
            self.config.scenarios = original_scenarios
        
        # Verify reports
        assert len(reports) == 1
        assert "baseline" in reports
        
        report = reports["baseline"]
        assert report.scenario_id == "baseline" 
        assert report.scenario_name == "Baseline Test"
        assert report.page_url == "https://example.com"
        assert report.page_title == "Test Page"
        assert len(report.cookies) == 1
    
    def test_get_execution_summary(self):
        """Test execution summary generation."""
        # Mock execution results
        result1 = ScenarioExecutionResult("baseline")
        result1.success = True
        result1.cookies = [Mock(), Mock()]  # 2 cookies
        result1.policy_issues = []
        result1.execution_time_ms = 1500
        
        result2 = ScenarioExecutionResult("gpc_on")
        result2.success = False
        result2.cookies = [Mock()]  # 1 cookie
        result2.policy_issues = [Mock()]  # 1 issue
        result2.execution_time_ms = 800
        result2.errors = ["Test error"]
        
        self.orchestrator.execution_results = {
            "baseline": result1,
            "gpc_on": result2
        }
        
        # Get summary
        summary = self.orchestrator.get_execution_summary()
        
        # Verify summary
        assert summary["total_scenarios"] == 2
        assert summary["successful_scenarios"] == 1
        assert summary["success_rate"] == 50.0
        assert summary["total_cookies_collected"] == 3
        assert summary["total_policy_issues"] == 1
        assert summary["total_execution_time_ms"] == 2300
        assert summary["average_execution_time_ms"] == 1150
        assert "gpc_on" in summary["scenarios_with_errors"]
    
    @pytest.mark.asyncio
    async def test_orchestrator_cleanup(self):
        """Test orchestrator cleanup."""
        # Mock GPC simulator with contexts
        self.orchestrator.gpc_simulator.injected_contexts = {"ctx1", "ctx2"}
        
        # Cleanup
        await self.orchestrator.cleanup()
        
        # Verify cleanup
        assert len(self.orchestrator.gpc_simulator.injected_contexts) == 0
    
    @pytest.mark.asyncio
    async def test_screenshot_capture(self):
        """Test screenshot capture functionality."""
        mock_page = Mock(spec=Page)
        mock_page.screenshot = AsyncMock()
        
        # Test successful screenshot
        screenshot_path = await self.orchestrator._take_screenshot(mock_page, "test_suffix")
        
        assert screenshot_path != ""
        assert "screenshot_test_suffix_" in screenshot_path
        mock_page.screenshot.assert_called_once()
        
        # Test screenshot failure
        mock_page.screenshot = AsyncMock(side_effect=Exception("Screenshot failed"))
        screenshot_path = await self.orchestrator._take_screenshot(mock_page, "test_suffix") 
        
        assert screenshot_path == ""  # Should return empty string on failure
    
    def test_orchestrator_representation(self):
        """Test orchestrator string representation.""" 
        self.orchestrator.execution_results = {"test1": Mock(), "test2": Mock()}
        self.config.scenarios = [Mock(), Mock(), Mock()]
        
        repr_str = repr(self.orchestrator)
        
        assert "ScenarioOrchestrator" in repr_str
        assert "scenarios=3" in repr_str
        assert "executed=2" in repr_str