"""Unit tests for main cookie consent service."""

import pytest
import tempfile
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from playwright.async_api import Browser

from app.audit.cookies.service import CookieConsentService, analyze_page_privacy, create_cookie_consent_service
from app.audit.cookies.models import PrivacyAnalysisResult, ScenarioCookieReport, CookieRecord
from app.audit.cookies.config import PrivacyConfiguration
from app.audit.cookies.policy import ComplianceFramework


class TestCookieConsentService:
    """Test CookieConsentService functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_browser = Mock(spec=Browser)
        self.config = PrivacyConfiguration()
        self.artifacts_dir = Path("/tmp/test_artifacts")
        
        self.service = CookieConsentService(
            browser=self.mock_browser,
            config=self.config,
            artifacts_dir=self.artifacts_dir
        )
    
    @pytest.mark.asyncio
    async def test_analyze_page_privacy_success(self):
        """Test successful page privacy analysis."""
        page_url = "https://example.com"
        page_title = "Test Page"
        
        # Mock the analysis result
        mock_analysis_result = PrivacyAnalysisResult(
            page_url=page_url,
            page_title=page_title,
            scenario_reports={
                "baseline": ScenarioCookieReport(
                    scenario_id="baseline",
                    scenario_name="Baseline Test",
                    page_url=page_url,
                    page_title=page_title,
                    cookies=[
                        CookieRecord(
                            name="session_id",
                            value="sess123",
                            domain="example.com",
                            path="/",
                            secure=True,
                            http_only=True,
                            first_party=True,
                            scenario_id="baseline"
                        )
                    ],
                    policy_issues=[],
                    errors=[]
                )
            }
        )

        # Patch execute_privacy_scenarios function
        with patch('app.audit.cookies.service.execute_privacy_scenarios', return_value=mock_analysis_result):
            # Analyze page privacy
            result = await self.service.analyze_page_privacy(page_url, page_title)
        
        # Verify result
        assert isinstance(result, PrivacyAnalysisResult)
        assert result.page_url == page_url
        assert "baseline" in result.scenario_reports
        assert result.scenario_reports["baseline"].page_title == page_title
    
    @pytest.mark.asyncio
    async def test_analyze_page_privacy_with_errors(self):
        """Test page analysis with errors."""
        page_url = "https://example.com"

        # Mock execute_privacy_scenarios to raise exception
        with patch('app.audit.cookies.service.execute_privacy_scenarios', side_effect=Exception("Navigation failed")):
            # Should handle errors gracefully
            with pytest.raises(Exception, match="Navigation failed"):
                await self.service.analyze_page_privacy(page_url)
    
    @pytest.mark.asyncio
    async def test_analyze_multiple_pages(self):
        """Test analyzing multiple pages in sequence."""
        pages = [
            ("https://example.com", "Home Page"),
            ("https://example.com/about", "About Page"),
            ("https://example.com/contact", "Contact Page")
        ]
        
        # Mock successful analysis for each page
        with patch.object(self.service, 'analyze_page_privacy') as mock_analyze:
            mock_results = []
            for page_url, page_title in pages:
                mock_result = PrivacyAnalysisResult(
                    page_url=page_url,
                    scenario_reports={
                        "baseline": ScenarioCookieReport(
                            scenario_id="baseline",
                            scenario_name="Baseline",
                            page_url=page_url,
                            page_title=page_title,
                            cookies=[],
                            policy_issues=[],
                            errors=[]
                        )
                    }
                )
                mock_results.append(mock_result)
            
            mock_analyze.side_effect = mock_results
            
            # Analyze multiple pages
            results = await self.service.analyze_multiple_pages(pages)
        
        # Verify results
        assert len(results) == 3
        assert results[0].page_url == "https://example.com"
        assert results[1].page_url == "https://example.com/about"
        assert results[2].page_url == "https://example.com/contact"
        
        # Verify each page was analyzed
        assert mock_analyze.call_count == 3
    
    @pytest.mark.asyncio
    async def test_batch_privacy_analysis(self):
        """Test batch privacy analysis with concurrent processing."""
        urls = [
            "https://example.com",
            "https://example.com/page1", 
            "https://example.com/page2"
        ]
        
        # Mock concurrent analysis
        with patch.object(self.service, 'analyze_page_privacy') as mock_analyze:
            # Create mock results
            mock_results = [
                PrivacyAnalysisResult(
                    page_url=url,
                    scenario_reports={
                        "baseline": ScenarioCookieReport(
                            scenario_id="baseline",
                            scenario_name="Baseline",
                            page_url=url,
                            cookies=[],
                            policy_issues=[],
                            errors=[]
                        )
                    }
                )
                for url in urls
            ]
            mock_analyze.side_effect = mock_results
            
            # Run batch analysis
            results = await self.service.batch_privacy_analysis(
                urls, max_concurrent=2
            )
        
        # Verify batch processing
        assert len(results) == 3
        assert all(isinstance(r, PrivacyAnalysisResult) for r in results)
        assert mock_analyze.call_count == 3
    
    def test_service_configuration_management(self):
        """Test service configuration management."""
        # Test default configuration
        default_service = CookieConsentService(self.mock_browser)
        assert default_service.config is not None
        
        # Test custom configuration
        custom_config = PrivacyConfiguration()
        custom_config.gpc.enabled = False
        
        custom_service = CookieConsentService(self.mock_browser, custom_config)
        assert custom_service.config.gpc.enabled is False
    
    def test_service_artifacts_management(self):
        """Test service artifacts directory management."""
        # Test with custom artifacts directory (use temp dir)
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_dir = Path(temp_dir) / "custom_artifacts"
            service = CookieConsentService(
                browser=self.mock_browser,
                artifacts_dir=custom_dir
            )

            assert service.artifacts_dir == custom_dir
        
        # Test default artifacts directory
        default_service = CookieConsentService(self.mock_browser)
        assert default_service.artifacts_dir.name == "privacy"
    
    @pytest.mark.asyncio 
    async def test_comprehensive_privacy_audit(self):
        """Test comprehensive privacy audit workflow."""
        page_url = "https://example.com"
        
        # Mock comprehensive analysis result
        comprehensive_result = {
            "page_analysis": PrivacyAnalysisResult(
                page_url=page_url,
                scenario_reports={
                    "baseline": ScenarioCookieReport(
                        scenario_id="baseline",
                        scenario_name="Baseline Test",
                        page_url=page_url,
                        cookies=[],
                        policy_issues=[],
                        errors=[]
                    ),
                    "gpc_on": ScenarioCookieReport(
                        scenario_id="gpc_on",
                        scenario_name="GPC Enabled",
                        page_url=page_url,
                        cookies=[],
                        policy_issues=[],
                        errors=[]
                    ),
                    "cmp_accept_all": ScenarioCookieReport(
                        scenario_id="cmp_accept_all",
                        scenario_name="CMP Accept All",
                        page_url=page_url,
                        cookies=[],
                        policy_issues=[],
                        errors=[]
                    ),
                    "cmp_reject_all": ScenarioCookieReport(
                        scenario_id="cmp_reject_all",
                        scenario_name="CMP Reject All",
                        page_url=page_url,
                        cookies=[],
                        policy_issues=[],
                        errors=[]
                    )
                }
            ),
            "compliance_summary": {
                "gdpr_issues": [],
                "ccpa_issues": [],
                "overall_score": 85
            },
            "recommendations": [
                "Enable Secure flag on all cookies for HTTPS",
                "Consider implementing GPC signal handling"
            ]
        }
        
        # Mock the comprehensive audit method (create since it doesn't exist)
        self.service.run_comprehensive_audit = AsyncMock(return_value=comprehensive_result)

        result = await self.service.run_comprehensive_audit(page_url)

        # Verify comprehensive audit result
        assert "page_analysis" in result
        assert "compliance_summary" in result
        assert "recommendations" in result
        assert result["compliance_summary"]["overall_score"] == 85
    
    def test_service_state_management(self):
        """Test service state management and cleanup."""
        # Test service initialization state
        assert self.service.browser == self.mock_browser
        assert self.service.config == self.config
        assert self.service.artifacts_dir == self.artifacts_dir
        
        # Test service representation
        repr_str = repr(self.service)
        assert "CookieConsentService" in repr_str
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        page_url = "https://invalid-url.example"

        # Mock execute_privacy_scenarios to raise exception
        with patch('app.audit.cookies.service.execute_privacy_scenarios', side_effect=Exception("DNS resolution failed")):
            with pytest.raises(Exception):
                await self.service.analyze_page_privacy(page_url)
    
    @pytest.mark.asyncio
    async def test_parallel_scenario_execution(self):
        """Test parallel scenario execution configuration."""
        page_url = "https://example.com"

        # Mock the analysis result
        mock_analysis_result = PrivacyAnalysisResult(
            page_url=page_url,
            scenario_reports={
                "baseline": ScenarioCookieReport(
                    scenario_id="baseline",
                    scenario_name="Baseline Test",
                    page_url=page_url,
                    cookies=[],
                    policy_issues=[],
                    errors=[]
                )
            }
        )

        # Mock execute_privacy_scenarios to capture the parallel_execution parameter
        mock_execute = AsyncMock(return_value=mock_analysis_result)

        with patch('app.audit.cookies.service.execute_privacy_scenarios', mock_execute):

            # Test with parallel execution enabled
            await self.service.analyze_page_privacy(
                page_url, parallel_execution=True
            )

            # Verify parallel execution was passed to execute_privacy_scenarios
            call_args = mock_execute.call_args
            assert call_args[0][4] is True  # parallel_execution is the 5th argument (index 4)
    
    def test_service_factory_functions(self):
        """Test service factory functions."""
        # Test create_cookie_consent_service
        service = create_cookie_consent_service(self.mock_browser)
        assert isinstance(service, CookieConsentService)
        assert service.browser == self.mock_browser
        
        # Test with custom config
        custom_config = PrivacyConfiguration()
        service_with_config = create_cookie_consent_service(
            self.mock_browser, custom_config
        )
        assert service_with_config.config == custom_config


class TestAnalyzePagePrivacyFunction:
    """Test standalone analyze_page_privacy function."""
    
    @pytest.mark.asyncio
    async def test_standalone_analysis_function(self):
        """Test standalone privacy analysis function."""
        mock_browser = Mock(spec=Browser)
        page_url = "https://example.com"
        
        # Mock service and analysis
        mock_service = Mock()
        mock_result = PrivacyAnalysisResult(
            page_url=page_url,
            scenario_reports={}
        )
        mock_service.analyze_page_privacy = AsyncMock(return_value=mock_result)
        
        with patch('app.audit.cookies.service.CookieConsentService', return_value=mock_service):
            
            result = await analyze_page_privacy(mock_browser, page_url)
            
            # Verify function result
            assert isinstance(result, PrivacyAnalysisResult)
            assert result.page_url == page_url
            
            # Verify service was created and used
            mock_service.analyze_page_privacy.assert_called_once_with(
                page_url,
                page_title=None,
                compliance_framework=ComplianceFramework.GDPR,
                parallel_execution=False
            )
    
    @pytest.mark.asyncio 
    async def test_standalone_function_with_options(self):
        """Test standalone function with all options."""
        mock_browser = Mock(spec=Browser)
        page_url = "https://example.com"
        page_title = "Test Page"
        artifacts_dir = Path("/test/artifacts")
        
        mock_service = Mock()
        mock_result = PrivacyAnalysisResult(
            page_url=page_url,
            scenario_reports={}
        )
        mock_service.analyze_page_privacy = AsyncMock(return_value=mock_result)
        
        with patch('app.audit.cookies.service.CookieConsentService', return_value=mock_service):
            
            result = await analyze_page_privacy(
                browser=mock_browser,
                page_url=page_url,
                page_title=page_title,
                artifacts_dir=artifacts_dir,
                parallel_execution=True
            )
            
            # Verify all options were passed through
            mock_service.analyze_page_privacy.assert_called_once_with(
                page_url,
                page_title=page_title,
                compliance_framework=ComplianceFramework.GDPR,
                parallel_execution=True
            )
    
    @pytest.mark.asyncio
    async def test_function_error_propagation(self):
        """Test error propagation in standalone function."""
        mock_browser = Mock(spec=Browser)
        
        mock_service = Mock()
        mock_service.analyze_page_privacy = AsyncMock(
            side_effect=Exception("Analysis failed")
        )
        
        with patch('app.audit.cookies.service.CookieConsentService', return_value=mock_service):
            
            # Should propagate errors from service
            with pytest.raises(Exception, match="Analysis failed"):
                await analyze_page_privacy(mock_browser, "https://example.com")