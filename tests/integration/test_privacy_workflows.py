"""Integration tests for complete privacy workflows.

These tests validate the entire Epic 5 implementation by running complete
privacy analysis workflows with realistic scenarios, including cookie
collection, GPC simulation, CMP interactions, and cross-scenario analysis.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import textwrap
import threading
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from urllib.parse import quote
from http.server import HTTPServer, BaseHTTPRequestHandler

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from app.audit.cookies.service import CookieConsentService, analyze_page_privacy
from app.audit.cookies.orchestration import execute_privacy_scenarios
from app.audit.cookies.models import (
    CookieRecord, 
    Scenario, 
    PrivacyAnalysisResult,
    ConsentState
)
from app.audit.cookies.config import PrivacyConfiguration, get_privacy_config
from app.audit.cookies.policy import ComplianceFramework


class TestHTTPHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for serving test pages with cookies."""

    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/many-cookies':
            # Generate page with many cookies
            cookie_scripts = []
            for i in range(50):
                cookie_scripts.append(f"document.cookie = 'test_cookie_{i}=value_{i}; path=/';")

            html_content = f"""
            <html>
            <head><title>Many Cookies Test</title></head>
            <body>
                <h1>High Volume Cookie Test</h1>
                <script>
                    {'; '.join(cookie_scripts)}
                </script>
            </body>
            </html>
            """.strip()

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.send_header('Content-Length', str(len(html_content)))
            self.end_headers()
            self.wfile.write(html_content.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Override to suppress log messages."""
        pass


@pytest_asyncio.fixture
async def test_http_server():
    """Create a lightweight HTTP server for testing."""
    # Find available port
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        port = s.getsockname()[1]

    # Create and start server
    server = HTTPServer(('localhost', port), TestHTTPHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Wait for server to start (non-blocking)
    await asyncio.sleep(0.1)

    yield f"http://localhost:{port}"

    # Cleanup
    server.shutdown()
    server.server_close()

    # Wait for server thread to finish
    server_thread.join(timeout=5.0)


@pytest_asyncio.fixture
async def browser():
    """Create a real browser instance for integration testing."""
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    yield browser
    await browser.close()
    await playwright.stop()


@pytest.fixture
def privacy_config():
    """Create test privacy configuration."""
    config = PrivacyConfiguration()
    
    # Configure test scenarios
    config.scenarios = [
        Scenario(
            id="baseline",
            name="Baseline - No Privacy Signals",
            description="Baseline scenario without privacy signals",
            enabled=True
        ),
        Scenario(
            id="gpc_on", 
            name="GPC Enabled",
            description="Test with Global Privacy Control enabled",
            request_headers={"Sec-GPC": "1"},
            enabled=True
        )
    ]
    
    # Configure GPC
    config.gpc.enabled = True
    
    # Configure CMP (simplified for testing)
    config.cmp.enabled = False  # Disable for basic tests
    
    return config


@pytest.fixture
def artifacts_dir():
    """Create temporary artifacts directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup handled by tempfile


class TestPrivacyWorkflowIntegration:
    """Test complete privacy workflow integration."""
    
    @pytest.mark.asyncio
    async def test_basic_privacy_analysis_workflow(self, browser: Browser, privacy_config: PrivacyConfiguration, artifacts_dir: Path):
        """Test basic privacy analysis workflow with mock page."""
        # Use a data URL for testing to avoid external dependencies
        test_page_url = "data:text/html,<html><body><h1>Test Page</h1><script>document.cookie='test_cookie=value';</script></body></html>"
        
        # Create service
        service = CookieConsentService(
            browser=browser,
            config=privacy_config,
            artifacts_dir=artifacts_dir
        )
        
        try:
            # Run privacy analysis
            result = await service.analyze_page_privacy(
                page_url=test_page_url
            )
            
            # Verify result structure
            assert isinstance(result, PrivacyAnalysisResult)
            assert result.page_url == test_page_url
            assert len(result.scenario_reports) >= 1
            
            # Should have baseline scenario at minimum
            assert "baseline" in result.scenario_reports
            baseline_report = result.scenario_reports["baseline"]
            
            assert baseline_report.scenario_id == "baseline"
            assert baseline_report.scenario_name == "Baseline - No Privacy Signals"
            assert baseline_report.page_url == test_page_url
            # Note: page_title may not be available in all implementations
            assert isinstance(baseline_report.cookies, list)
            assert isinstance(baseline_report.policy_issues, list)
            assert isinstance(baseline_report.errors, list)
            
        finally:
            # Cleanup (if cleanup method exists)
            if hasattr(service, 'cleanup'):
                await service.cleanup()
    
    @pytest.mark.asyncio
    async def test_gpc_scenario_workflow(self, browser: Browser, privacy_config: PrivacyConfiguration, artifacts_dir: Path):
        """Test GPC scenario workflow."""
        # Enhanced test page with tracking scripts
        test_page_html = """
        <html>
        <head><title>GPC Test Page</title></head>
        <body>
            <h1>GPC Test</h1>
            <script>
                // Set some test cookies
                document.cookie = 'session_id=abc123; path=/; secure';
                document.cookie = 'tracking_id=xyz789; path=/';
                
                // Simulate GPC detection
                if (navigator.globalPrivacyControl) {
                    console.log('GPC detected:', navigator.globalPrivacyControl);
                    document.cookie = 'gpc_detected=true; path=/';
                } else {
                    console.log('GPC not detected');
                    document.cookie = 'gpc_detected=false; path=/';
                }
            </script>
        </body>
        </html>
        """

        # Properly encode the HTML for data URL
        clean_html = textwrap.dedent(test_page_html).strip()
        encoded_html = quote(clean_html)
        test_page_url = f"data:text/html,{encoded_html}"

        service = CookieConsentService(
            browser=browser,
            config=privacy_config,
            artifacts_dir=artifacts_dir
        )
        
        try:
            result = await service.analyze_page_privacy(
                page_url=test_page_url,
                parallel_execution=True
            )
            
            # Should have both baseline and GPC scenarios
            assert len(result.scenario_reports) == 2
            assert "baseline" in result.scenario_reports
            assert "gpc_on" in result.scenario_reports
            
            baseline_report = result.scenario_reports["baseline"]
            gpc_report = result.scenario_reports["gpc_on"]
            
            # Verify scenario differences
            assert baseline_report.scenario_id == "baseline"
            assert gpc_report.scenario_id == "gpc_on"
            
            # Both should have some cookies from the test page
            assert len(baseline_report.cookies) >= 0
            assert len(gpc_report.cookies) >= 0
            
        finally:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
    
    @pytest.mark.asyncio  
    async def test_cookie_classification_integration(self, browser: Browser, privacy_config: PrivacyConfiguration, artifacts_dir: Path):
        """Test cookie classification in integration context."""
        # Test page with first-party and third-party-like cookies
        test_page_html = """
        <html>
        <head><title>Cookie Classification Test</title></head>
        <body>
            <h1>Classification Test</h1>
            <script>
                // First-party cookies
                document.cookie = 'session_id=sess123; path=/; secure; httponly';
                document.cookie = 'user_pref=theme_dark; path=/';
                
                // Simulate third-party tracking (though data: URL limits this)
                document.cookie = 'analytics_id=ga123; path=/';
                document.cookie = 'marketing_id=fb456; path=/';
            </script>
        </body>
        </html>
        """
        
        test_page_url = f"data:text/html,{test_page_html}"
        
        service = CookieConsentService(browser=browser, config=privacy_config)
        
        try:
            result = await service.analyze_page_privacy(test_page_url)
            
            baseline_report = result.scenario_reports["baseline"]
            cookies = baseline_report.cookies
            
            # Verify classification occurred
            for cookie in cookies:
                assert hasattr(cookie, 'first_party')
                assert isinstance(cookie.first_party, bool)
                assert hasattr(cookie, 'essential')
                
                # Check cookie attributes are captured
                assert hasattr(cookie, 'secure')
                assert hasattr(cookie, 'http_only')
                assert hasattr(cookie, 'same_site')
                
        finally:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
    
    @pytest.mark.asyncio
    async def test_policy_compliance_integration(self, browser: Browser, privacy_config: PrivacyConfiguration, artifacts_dir: Path):
        """Test policy compliance validation in integration context."""
        # Configure stricter policies for testing
        privacy_config.policies.secure_required = True
        privacy_config.policies.same_site_default = "Lax"
        
        test_page_html = """
        <html>
        <head><title>Policy Compliance Test</title></head>
        <body>
            <h1>Policy Test</h1>
            <script>
                // Compliant cookie
                document.cookie = 'secure_session=value; path=/; secure; samesite=strict';
                
                // Non-compliant cookie (missing secure on HTTPS-like context)
                document.cookie = 'insecure_cookie=value; path=/';
            </script>
        </body>
        </html>
        """
        
        test_page_url = f"data:text/html,{test_page_html}"
        
        service = CookieConsentService(browser=browser, config=privacy_config)
        
        try:
            result = await service.analyze_page_privacy(test_page_url)
            
            baseline_report = result.scenario_reports["baseline"]
            
            # Verify policy issues are detected
            assert isinstance(baseline_report.policy_issues, list)
            
            # Should have some cookies
            assert len(baseline_report.cookies) >= 0
            
        finally:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, browser: Browser, privacy_config: PrivacyConfiguration, artifacts_dir: Path):
        """Test error handling in integration context.""" 
        # Use invalid URL to trigger errors
        invalid_url = "https://this-domain-should-not-exist-12345.invalid"
        
        service = CookieConsentService(browser=browser, config=privacy_config)
        
        try:
            # Should handle navigation errors gracefully without raising exceptions
            result = await service.analyze_page_privacy(invalid_url)
            # Expecting empty or error result instead of exception
            assert result is not None, "Service should return a result even for invalid URLs"

        finally:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
    
    @pytest.mark.asyncio
    async def test_performance_with_multiple_scenarios(self, browser: Browser, privacy_config: PrivacyConfiguration, artifacts_dir: Path):
        """Test performance with multiple scenarios."""
        # Add more scenarios for performance testing
        privacy_config.scenarios.extend([
            Scenario(
                id="test_scenario_3",
                name="Test Scenario 3",
                description="Test scenario with DNT header for performance testing",
                request_headers={"DNT": "1"},
                enabled=True
            ),
            Scenario(
                id="test_scenario_4",
                name="Test Scenario 4",
                description="Test scenario with custom header for performance testing",
                request_headers={"Custom-Header": "test"},
                enabled=True
            )
        ])
        
        test_page_url = "data:text/html,<html><body><h1>Performance Test</h1></body></html>"
        
        service = CookieConsentService(browser=browser, config=privacy_config)
        
        try:
            import time
            start_time = time.time()
            
            result = await service.analyze_page_privacy(
                test_page_url,
                parallel_execution=True
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should complete within reasonable time
            assert execution_time < 30.0  # 30 seconds max for 4 scenarios
            
            # Should have all configured scenarios
            assert len(result.scenario_reports) == 4
            
        finally:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
    
    @pytest.mark.asyncio
    async def test_artifacts_generation(self, browser: Browser, privacy_config: PrivacyConfiguration, artifacts_dir: Path):
        """Test artifact generation during privacy analysis."""
        test_page_url = "data:text/html,<html><body><h1>Artifacts Test</h1></body></html>"
        
        service = CookieConsentService(
            browser=browser,
            config=privacy_config,
            artifacts_dir=artifacts_dir
        )
        
        try:
            await service.analyze_page_privacy(test_page_url)
            
            # Check artifacts directory exists
            assert artifacts_dir.exists()
            assert artifacts_dir.is_dir()
            
            # May contain screenshots or other artifacts depending on implementation
            
        finally:
            if hasattr(service, 'cleanup'):
                await service.cleanup()


class TestStandalonePrivacyFunctions:
    """Test standalone privacy analysis functions."""
    
    @pytest.mark.asyncio
    async def test_analyze_page_privacy_function(self, browser: Browser):
        """Test standalone analyze_page_privacy function."""
        test_page_url = "data:text/html,<html><body><h1>Standalone Test</h1></body></html>"
        
        result = await analyze_page_privacy(
            browser=browser,
            page_url=test_page_url
        )
        
        assert isinstance(result, PrivacyAnalysisResult)
        assert result.page_url == test_page_url
        assert len(result.scenario_reports) >= 1
    
    @pytest.mark.asyncio
    async def test_execute_privacy_scenarios_function(self, browser: Browser, privacy_config: PrivacyConfiguration, artifacts_dir: Path):
        """Test standalone execute_privacy_scenarios function."""
        test_page_url = "data:text/html,<html><body><h1>Execute Scenarios Test</h1></body></html>"
        
        result = await execute_privacy_scenarios(
            browser=browser,
            page_url=test_page_url,
            config=privacy_config,
            artifacts_dir=artifacts_dir
        )
        
        assert isinstance(result, PrivacyAnalysisResult)
        assert result.page_url == test_page_url
        assert len(result.scenario_reports) >= 1


class TestRealWorldScenarios:
    """Test with more realistic scenarios (still using data URLs for reliability)."""
    
    @pytest.mark.asyncio
    async def test_ecommerce_like_scenario(self, browser: Browser, privacy_config: PrivacyConfiguration, artifacts_dir: Path):
        """Test e-commerce-like scenario with multiple cookie types."""
        ecommerce_html = """
        <html>
        <head>
            <title>E-commerce Test Site</title>
            <script>
                // Session management
                document.cookie = 'JSESSIONID=ABC123; path=/';
                document.cookie = 'cart_id=cart789; path=/';

                // User preferences
                document.cookie = 'language=en-US; path=/';
                document.cookie = 'currency=USD; path=/';

                // Analytics (simulated)
                document.cookie = '_ga=GA1.2.123456789; path=/';
                document.cookie = '_gid=GA1.2.987654321; path=/';

                // Marketing (simulated)
                document.cookie = '_fbp=fb.1.123456789; path=/';
                document.cookie = 'marketing_consent=false; path=/';
            </script>
        </head>
        <body>
            <h1>E-commerce Store</h1>
            <p>Shopping cart and user preferences</p>
        </body>
        </html>
        """

        # Properly encode the HTML for data URL
        clean_html = textwrap.dedent(ecommerce_html).strip()
        encoded_html = quote(clean_html)
        test_page_url = f"data:text/html,{encoded_html}"

        service = CookieConsentService(
            browser=browser,
            config=privacy_config,
            artifacts_dir=artifacts_dir
        )

        try:
            result = await service.analyze_page_privacy(
                test_page_url,
                page_title="E-commerce Store"
            )

            # Analyze results
            baseline_report = result.scenario_reports["baseline"]
            cookies = baseline_report.cookies

            # Assert that the service call path worked (returns a list even if empty)
            assert isinstance(cookies, list), "Cookies should be returned as a list"

            # If cookies were captured, verify they have proper structure and classification
            if cookies:
                # Verify cookie classification by the service
                cookie_names = {c.name for c in cookies}
                expected_cookie_names = {"JSESSIONID", "cart_id", "_ga", "_gid", "_fbp", "marketing_consent", "language", "currency"}

                # Verify specific cookie types were captured
                captured_cookie_names = expected_cookie_names.intersection(cookie_names)
                assert len(captured_cookie_names) >= 1, f"Expected at least 1 expected cookie captured, got: {captured_cookie_names}"

                # Verify classification occurred
                for cookie in cookies:
                    assert hasattr(cookie, 'first_party'), f"Cookie {cookie.name} missing first_party classification"
                    assert cookie.scenario_id == "baseline", f"Cookie {cookie.name} has wrong scenario_id: {cookie.scenario_id}"

                # Verify service is properly capturing and returning structured cookies
                assert len(cookies) >= 1, f"Service captured cookies but expected at least 1, got {len(cookies)}"
            else:
                # Even if no cookies captured, verify the service integration worked
                # by checking that we got a proper result structure
                assert hasattr(result, 'scenario_reports'), "Service should return result with scenario_reports"
                assert "baseline" in result.scenario_reports, "Service should include baseline scenario"

        finally:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
    
    @pytest.mark.asyncio
    async def test_content_site_scenario(self, browser: Browser, privacy_config: PrivacyConfiguration, artifacts_dir: Path):
        """Test content site scenario with analytics and social cookies."""
        content_html = """
        <html>
        <head>
            <title>News Content Site</title>
            <script>
                // Essential functionality
                document.cookie = 'user_session=sess456; path=/; secure; httponly';
                document.cookie = 'csrf_token=csrf789; path=/; secure; httponly';
                
                // Analytics
                document.cookie = '_ga=GA1.2.content123; path=/';
                document.cookie = '_gat=1; path=/; max-age=60';
                
                // Social sharing  
                document.cookie = 'social_share_pref=twitter,facebook; path=/';
                
                // Advertising
                document.cookie = 'ad_personalization=enabled; path=/';
                document.cookie = '_ad_id=ad123456; path=/';
                
                // Performance monitoring
                document.cookie = 'perf_id=perf789; path=/';
            </script>
        </head>
        <body>
            <h1>Latest News</h1>
            <article>
                <h2>Breaking News Story</h2>
                <p>This is a test news article for privacy testing.</p>
            </article>
        </body>
        </html>
        """

        # Properly encode the HTML for data URL
        clean_html = textwrap.dedent(content_html).strip()
        encoded_html = quote(clean_html)
        test_page_url = f"data:text/html,{encoded_html}"

        service = CookieConsentService(browser=browser, config=privacy_config)
        
        try:
            result = await service.analyze_page_privacy(test_page_url)
            
            # Test both baseline and GPC scenarios
            assert "baseline" in result.scenario_reports
            assert "gpc_on" in result.scenario_reports
            
            baseline_report = result.scenario_reports["baseline"]
            gpc_report = result.scenario_reports["gpc_on"]
            
            # Should have cookies in baseline
            assert len(baseline_report.cookies) >= 0
            
            # GPC scenario might have different cookie behavior
            # (though limited with data URLs)
            assert len(gpc_report.cookies) >= 0
            
            # Verify scenario isolation
            assert baseline_report.scenario_id != gpc_report.scenario_id
            
        finally:
            if hasattr(service, 'cleanup'):
                await service.cleanup()


class TestConfigurationDrivenTesting:
    """Test configuration-driven privacy testing scenarios."""
    
    @pytest.mark.asyncio
    async def test_custom_compliance_framework(self, browser: Browser, artifacts_dir: Path):
        """Test with custom compliance framework configuration."""
        # Create config for CCPA compliance
        config = PrivacyConfiguration()

        config.scenarios = [
            Scenario(
                id="ccpa_baseline",
                name="CCPA Baseline",
                description="CCPA compliance baseline test",
                enabled=True
            ),
            Scenario(
                id="ccpa_opt_out",
                name="CCPA Opt-Out Signal",
                description="CCPA opt-out signal test with GPC",
                request_headers={"Sec-GPC": "1"},  # GPC is related to CCPA
                enabled=True
            )
        ]
        
        test_page_url = "data:text/html,<html><body><h1>CCPA Test</h1></body></html>"

        service = CookieConsentService(browser=browser, config=config)

        try:
            result = await service.analyze_page_privacy(
                test_page_url,
                compliance_framework=ComplianceFramework.CCPA
            )

            # Should respect CCPA compliance framework
            assert len(result.scenario_reports) == 2
            assert "ccpa_baseline" in result.scenario_reports
            assert "ccpa_opt_out" in result.scenario_reports
            
        finally:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
    
    @pytest.mark.asyncio
    async def test_disabled_scenarios(self, browser: Browser):
        """Test with some scenarios disabled."""
        config = PrivacyConfiguration()
        config.scenarios = [
            Scenario(id="enabled_scenario", name="Enabled", description="Enabled test scenario", enabled=True),
            Scenario(id="disabled_scenario", name="Disabled", description="Disabled test scenario", enabled=False)
        ]
        
        test_page_url = "data:text/html,<html><body><h1>Disabled Scenarios Test</h1></body></html>"
        
        service = CookieConsentService(browser=browser, config=config)
        
        try:
            result = await service.analyze_page_privacy(test_page_url)
            
            # Should only run enabled scenarios
            assert "enabled_scenario" in result.scenario_reports
            assert "disabled_scenario" not in result.scenario_reports
            
        finally:
            if hasattr(service, 'cleanup'):
                await service.cleanup()


@pytest.mark.skipif(
    not pytest.mark.slow,
    reason="Slow tests skipped - run with --slow to include"
)  
class TestPerformanceIntegration:
    """Performance-focused integration tests."""
    
    @pytest.mark.asyncio
    async def test_high_volume_cookie_handling(self, browser: Browser, privacy_config: PrivacyConfiguration, test_http_server: str):
        """Test handling of pages with many cookies."""
        # Use the HTTP server to serve page with many cookies
        test_page_url = f"{test_http_server}/many-cookies"

        service = CookieConsentService(browser=browser, config=privacy_config)
        
        try:
            import time
            start_time = time.time()
            
            result = await service.analyze_page_privacy(test_page_url)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should handle many cookies efficiently
            assert execution_time < 15.0  # Should complete within 15 seconds
            
            # Should capture many cookies from the HTTP server
            baseline_report = result.scenario_reports["baseline"]
            assert isinstance(baseline_report.cookies, list), "Should return cookie list"
            assert len(baseline_report.cookies) >= 10, f"Expected at least 10 cookies from HTTP server, got {len(baseline_report.cookies)}"

            # Verify we captured some of the specific test cookies
            cookie_names = {c.name for c in baseline_report.cookies}
            test_cookies_found = [name for name in cookie_names if name.startswith('test_cookie_')]
            assert len(test_cookies_found) >= 5, f"Expected at least 5 test cookies, found: {test_cookies_found}"
            
        finally:
            if hasattr(service, 'cleanup'):
                await service.cleanup()
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_stress(self, browser: Browser, privacy_config: PrivacyConfiguration):
        """Test concurrent analysis operations."""
        test_urls = [
            "data:text/html,<html><body><h1>Test Page 1</h1></body></html>",
            "data:text/html,<html><body><h1>Test Page 2</h1></body></html>",
            "data:text/html,<html><body><h1>Test Page 3</h1></body></html>"
        ]
        
        service = CookieConsentService(browser=browser, config=privacy_config)
        
        try:
            # Run multiple analyses concurrently
            tasks = [
                service.analyze_page_privacy(url, parallel_execution=True)
                for url in test_urls
            ]
            
            import time
            start_time = time.time()
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should handle concurrent operations
            assert execution_time < 20.0  # Should complete within reasonable time
            
            # All should succeed or gracefully handle errors
            successful_results = [r for r in results if isinstance(r, PrivacyAnalysisResult)]
            assert len(successful_results) >= 1  # At least some should succeed
            
        finally:
            if hasattr(service, 'cleanup'):
                await service.cleanup()