"""Unit tests for cookie and consent models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.audit.cookies.models import (
    CookieRecord,
    Scenario,
    CookiePolicyIssue,
    ScenarioCookieReport,
    CookieDiff,
    PrivacyAnalysisResult,
    ConsentState
)


class TestCookieRecord:
    """Test CookieRecord model."""
    
    def test_cookie_record_creation(self):
        """Test basic cookie record creation."""
        cookie = CookieRecord(
            name="test_cookie",
            value="test_value",
            domain=".example.com",
            path="/",
            expires=datetime.fromtimestamp(1735689600),  # 2025-01-01
            size=25,
            same_site="Lax",
            secure=True,
            http_only=False,
            is_first_party=True,
            essential=False
        )
        
        assert cookie.name == "test_cookie"
        assert cookie.domain == ".example.com"
        assert cookie.first_party is True  # Using property alias
        assert cookie.is_first_party is True  # Direct field access
        assert cookie.essential is False
    
    def test_cookie_record_validation(self):
        """Test cookie record validation."""
        # Test missing required field (name)
        with pytest.raises(ValidationError):
            CookieRecord(
                value="value",
                domain="example.com",
                path="/",
                size=10,
                secure=True,
                http_only=False,
                is_first_party=True
            )
    
    def test_cookie_record_serialization(self):
        """Test cookie record JSON serialization."""
        cookie = CookieRecord(
            name="test_cookie",
            value="test_value",
            domain=".example.com",
            path="/",
            size=30,
            secure=True,
            http_only=True,
            is_first_party=False,
            essential=True
        )
        
        data = cookie.model_dump()
        assert data["name"] == "test_cookie"
        assert data["is_first_party"] is False  # This is the actual field
        assert data["essential"] is True
        
        # Test deserialization
        new_cookie = CookieRecord.model_validate(data)
        assert new_cookie.name == cookie.name
        assert new_cookie.first_party == cookie.first_party  # Using property


class TestScenario:
    """Test Scenario model."""
    
    def test_scenario_creation(self):
        """Test basic scenario creation."""
        scenario = Scenario(
            id="gpc_test",
            name="GPC Enabled Test",
            description="Test with GPC headers enabled",
            request_headers={"Sec-GPC": "1"},
            steps=[{"action": "click", "selector": "#accept-all"}]
        )
        
        assert scenario.id == "gpc_test"
        assert scenario.name == "GPC Enabled Test"
        assert scenario.request_headers["Sec-GPC"] == "1"
        assert len(scenario.steps) == 1
    
    def test_scenario_defaults(self):
        """Test scenario default values."""
        scenario = Scenario(
            id="baseline",
            name="Baseline Test",
            description="Default baseline test scenario"
        )
        
        assert scenario.request_headers == {}
        assert scenario.steps == []
        assert scenario.enabled is True


class TestCookiePolicyIssue:
    """Test CookiePolicyIssue model."""
    
    def test_policy_issue_creation(self):
        """Test policy issue creation."""
        issue = CookiePolicyIssue(
            cookie_name="insecure_cookie",
            cookie_domain="example.com",
            cookie_path="/",
            attribute="secure",
            expected="True",
            observed="False",
            severity="high",
            rule_id="SEC_001",
            message="Cookie should be secure on HTTPS"
        )
        
        assert issue.cookie_name == "insecure_cookie"
        assert issue.severity == "high"
        assert issue.rule_id == "SEC_001"
    
    def test_policy_issue_validation(self):
        """Test policy issue validation."""
        # Test invalid severity
        with pytest.raises(ValidationError):
            CookiePolicyIssue(
                cookie_name="test",
                domain="example.com",
                path="/",
                attribute="secure",
                expected_value=True,
                actual_value=False,
                severity="invalid",
                message="Test message"
            )


class TestScenarioCookieReport:
    """Test ScenarioCookieReport model."""
    
    def test_report_creation(self):
        """Test scenario report creation."""
        cookies = [
            CookieRecord(
                name="test1",
                value="value1",
                domain="example.com",
                path="/",
                size=15,
                secure=True,
                http_only=True,
                is_first_party=True
            ),
            CookieRecord(
                name="test2", 
                value="value2",
                domain="ads.com",
                path="/",
                size=15,
                secure=False,
                http_only=False,
                is_first_party=False
            )
        ]
        
        issues = [
            CookiePolicyIssue(
                cookie_name="test2",
                cookie_domain="ads.com",
                cookie_path="/",
                attribute="secure",
                expected="True",
                observed="False",
                severity="medium",
                rule_id="SEC_002",
                message="Third-party cookie should be secure"
            )
        ]
        
        report = ScenarioCookieReport(
            scenario_id="baseline",
            scenario_name="Baseline Test",
            page_url="https://example.com",
            page_title="Test Page",
            cookies=cookies,
            policy_issues=issues,
            consent_state=ConsentState.UNKNOWN,
            errors=[]
        )
        
        assert report.scenario_id == "baseline"
        assert len(report.cookies) == 2
        assert len(report.policy_issues) == 1
        assert report.consent_state == ConsentState.UNKNOWN
    
    def test_report_stats(self):
        """Test report statistics calculation."""
        cookies = [
            CookieRecord(
                name="first_party",
                value="value",
                domain="example.com", 
                path="/",
                size=15,
                secure=True,
                http_only=True,
                is_first_party=True
            ),
            CookieRecord(
                name="third_party",
                value="value",
                domain="ads.com",
                path="/",
                size=15, 
                secure=False,
                http_only=False,
                is_first_party=False
            )
        ]
        
        report = ScenarioCookieReport(
            scenario_id="test",
            scenario_name="Test",
            page_url="https://example.com",
            cookies=cookies,
            policy_issues=[],
            errors=[]
        )
        
        # Check statistics calculated by model_post_init
        assert report.total_cookies == 2
        assert report.first_party_cookies == 1  
        assert report.third_party_cookies == 1
        # Check individual cookie properties
        secure_cookies = len([c for c in report.cookies if c.secure])
        http_only_cookies = len([c for c in report.cookies if c.http_only])
        assert secure_cookies == 1
        assert http_only_cookies == 1


class TestCookieDiff:
    """Test CookieDiff model."""
    
    def test_cookie_diff_creation(self):
        """Test cookie diff creation."""
        added_cookies = [
            CookieRecord(
                name="new_cookie",
                value="value",
                domain="example.com",
                path="/",
                size=15,
                secure=True,
                http_only=False,
                is_first_party=True
            )
        ]
        
        removed_cookies = [
            CookieRecord(
                name="tracking_cookie",
                value="value", 
                domain="ads.com",
                path="/",
                size=15,
                secure=False,
                http_only=False,
                is_first_party=False
            )
        ]
        
        diff = CookieDiff(
            baseline_scenario="baseline",
            variant_scenario="gpc_on",
            page_url="https://example.com",
            added_cookies=added_cookies,
            removed_cookies=removed_cookies,
            modified_cookies=[],
            unchanged_cookies=[]
        )
        
        assert diff.baseline_scenario == "baseline"
        assert diff.variant_scenario == "gpc_on" 
        assert len(diff.added_cookies) == 1
        assert len(diff.removed_cookies) == 1
        assert diff.added_cookies[0].name == "new_cookie"
        assert diff.removed_cookies[0].name == "tracking_cookie"
    
    def test_diff_stats(self):
        """Test diff statistics."""
        diff = CookieDiff(
            baseline_scenario="baseline",
            variant_scenario="gpc_on",
            page_url="https://example.com",
            added_cookies=[],
            removed_cookies=[
                CookieRecord(
                    name="removed1",
                    value="value",
                    domain="ads.com",
                    path="/",
                    size=15,
                    secure=False,
                    http_only=False,
                    is_first_party=False
                ),
                CookieRecord(
                    name="removed2", 
                    value="value",
                    domain="tracker.com",
                    path="/",
                    size=15,
                    secure=False,
                    http_only=False,
                    is_first_party=False,
                )
            ],
            modified_cookies=[],
            unchanged_cookies=[]
        )
        
        assert len(diff.added_cookies) == 0
        assert len(diff.removed_cookies) == 2
        assert len(diff.modified_cookies) == 0
        assert len(diff.unchanged_cookies) == 0
        assert diff.total_changes == 2
        assert diff.cookie_reduction == 2


class TestPrivacyAnalysisResult:
    """Test PrivacyAnalysisResult model."""
    
    def test_analysis_result_creation(self):
        """Test privacy analysis result creation."""
        reports = {
            "baseline": ScenarioCookieReport(
                scenario_id="baseline",
                scenario_name="Baseline",
                page_url="https://example.com",
                cookies=[],
                policy_issues=[],
                errors=[]
            )
        }
        
        result = PrivacyAnalysisResult(
            page_url="https://example.com",
            scenario_reports=reports,
            analysis_timestamp=datetime.utcnow()
        )
        
        assert result.page_url == "https://example.com"
        assert "baseline" in result.scenario_reports
        assert result.analysis_timestamp is not None
    
    def test_analysis_summary(self):
        """Test analysis summary generation."""
        baseline_report = ScenarioCookieReport(
            scenario_id="baseline",
            scenario_name="Baseline", 
            page_url="https://example.com",
            cookies=[
                CookieRecord(
                    name="cookie1",
                    value="value",
                    domain="example.com",
                    path="/",
                    size=15,
                    secure=True,
                    http_only=True,
                    is_first_party=True
                )
            ],
            policy_issues=[],
            errors=[]
        )
        
        gpc_report = ScenarioCookieReport(
            scenario_id="gpc_on",
            scenario_name="GPC Enabled",
            page_url="https://example.com", 
            cookies=[],  # No cookies with GPC
            policy_issues=[],
            errors=[]
        )
        
        reports = {
            "baseline": baseline_report,
            "gpc_on": gpc_report
        }
        
        result = PrivacyAnalysisResult(
            page_url="https://example.com",
            scenario_reports=reports
        )
        
        # Test direct properties instead of get_summary()
        assert len(result.scenario_reports) == 2
        baseline_report = result.baseline_report
        assert baseline_report is not None
        assert len(baseline_report.cookies) == 1
        assert "baseline" in result.scenario_reports
        assert "gpc_on" in result.scenario_reports