"""Unit tests for cookie policy compliance engine."""

import pytest
from datetime import datetime, timedelta

from app.audit.cookies.policy import PolicyComplianceEngine, ComplianceFramework
from app.audit.cookies.models import CookieRecord, CookiePolicyIssue
from app.audit.cookies.config import PrivacyConfiguration


class TestPolicyComplianceEngine:
    """Test policy compliance engine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PrivacyConfiguration()
        self.engine = PolicyComplianceEngine(self.config)
    
    def test_secure_cookie_validation_https(self):
        """Test secure cookie validation for HTTPS sites."""
        https_url = "https://example.com"
        
        # Secure cookie - should pass
        secure_cookie = CookieRecord(
            name="secure_session",
            value="abc123",
            domain="example.com",
            path="/",
            secure=True,
            http_only=True,
            first_party=True,
            scenario_id="test"
        )
        
        # Insecure cookie - should fail on HTTPS
        insecure_cookie = CookieRecord(
            name="insecure_cookie",
            value="xyz789",
            domain="example.com", 
            path="/",
            secure=False,
            http_only=False,
            first_party=True,
            scenario_id="test"
        )
        
        cookies = [secure_cookie, insecure_cookie]
        issues = self.engine.validate_cookie_policy(
            cookies, https_url, ComplianceFramework.GDPR, "test", "production"
        )
        
        # Should have issue for insecure cookie
        secure_issues = [issue for issue in issues if "secure" in issue.message.lower()]
        assert len(secure_issues) == 1
        assert secure_issues[0].cookie_name == "insecure_cookie"
        assert secure_issues[0].severity == "medium"
    
    def test_secure_cookie_validation_http(self):
        """Test secure cookie validation for HTTP sites."""
        http_url = "http://example.com"
        
        # Insecure cookie on HTTP - should be acceptable
        insecure_cookie = CookieRecord(
            name="session_id",
            value="abc123", 
            domain="example.com",
            path="/",
            secure=False,
            http_only=True,
            first_party=True,
            scenario_id="test"
        )
        
        cookies = [insecure_cookie]
        issues = self.engine.validate_cookie_policy(
            cookies, http_url, ComplianceFramework.GDPR, "test", "development"
        )
        
        # Should have no secure-related issues for HTTP
        secure_issues = [issue for issue in issues if "secure" in issue.message.lower()]
        assert len(secure_issues) == 0
    
    def test_http_only_validation(self):
        """Test HttpOnly attribute validation."""
        page_url = "https://example.com"
        
        # Session cookie without HttpOnly - should fail
        session_cookie = CookieRecord(
            name="PHPSESSID",
            value="session123",
            domain="example.com",
            path="/", 
            secure=True,
            http_only=False,  # Should be True for session cookies
            first_party=True,
            essential=True,
            scenario_id="test"
        )
        
        # Non-session cookie without HttpOnly - acceptable
        preference_cookie = CookieRecord(
            name="language",
            value="en-US",
            domain="example.com",
            path="/",
            secure=True, 
            http_only=False,
            first_party=True,
            essential=False,
            scenario_id="test"
        )
        
        cookies = [session_cookie, preference_cookie]
        issues = self.engine.validate_cookie_policy(
            cookies, page_url, ComplianceFramework.GDPR, "test", "production"
        )
        
        # Should have HttpOnly issue for session cookie only
        http_only_issues = [issue for issue in issues if "httponly" in issue.message.lower()]
        assert len(http_only_issues) == 1
        assert http_only_issues[0].cookie_name == "PHPSESSID"
    
    def test_same_site_validation(self):
        """Test SameSite attribute validation."""
        page_url = "https://example.com"
        
        # Cookies with various SameSite values
        none_cookie = CookieRecord(
            name="cross_site_cookie",
            value="value1",
            domain="example.com",
            path="/",
            secure=True,  # Required for SameSite=None
            http_only=False,
            same_site="None",
            first_party=True,
            scenario_id="test"
        )
        
        lax_cookie = CookieRecord(
            name="normal_cookie", 
            value="value2",
            domain="example.com",
            path="/",
            secure=True,
            http_only=False,
            same_site="Lax",
            first_party=True,
            scenario_id="test"
        )
        
        strict_cookie = CookieRecord(
            name="sensitive_cookie",
            value="value3",
            domain="example.com",
            path="/",
            secure=True,
            http_only=True,
            same_site="Strict",
            first_party=True,
            scenario_id="test"
        )
        
        # Cookie with SameSite=None but not Secure
        invalid_none_cookie = CookieRecord(
            name="invalid_cookie",
            value="value4",
            domain="example.com",
            path="/",
            secure=False,  # Invalid with SameSite=None
            http_only=False,
            same_site="None", 
            first_party=True,
            scenario_id="test"
        )
        
        cookies = [none_cookie, lax_cookie, strict_cookie, invalid_none_cookie]
        issues = self.engine.validate_cookie_policy(
            cookies, page_url, ComplianceFramework.GDPR, "test", "production"
        )
        
        # Should have issue for SameSite=None without Secure
        samesite_issues = [issue for issue in issues if "samesite" in issue.message.lower()]
        assert len(samesite_issues) == 1
        assert samesite_issues[0].cookie_name == "invalid_cookie"
    
    def test_gdpr_compliance_validation(self):
        """Test GDPR-specific compliance validation."""
        page_url = "https://example.com"
        
        # Non-essential third-party cookie - requires consent under GDPR
        tracking_cookie = CookieRecord(
            name="_ga",
            value="GA1.2.123456789",
            domain="google-analytics.com",
            path="/",
            secure=False,
            http_only=False,
            first_party=False,
            essential=False,
            scenario_id="test"
        )
        
        # Essential first-party cookie - allowed under GDPR
        session_cookie = CookieRecord(
            name="session_id",
            value="session123",
            domain="example.com", 
            path="/",
            secure=True,
            http_only=True,
            first_party=True,
            essential=True,
            scenario_id="test"
        )
        
        cookies = [tracking_cookie, session_cookie]
        issues = self.engine.validate_cookie_policy(
            cookies, page_url, ComplianceFramework.GDPR, "test", "production"
        )
        
        # Should flag non-essential third-party cookie
        consent_issues = [issue for issue in issues if "consent" in issue.message.lower()]
        assert len(consent_issues) >= 1
        
        # Find the tracking cookie issue
        tracking_issues = [issue for issue in issues if issue.cookie_name == "_ga"]
        assert len(tracking_issues) >= 1
    
    def test_ccpa_compliance_validation(self):
        """Test CCPA-specific compliance validation."""
        page_url = "https://example.com"
        
        # Third-party cookie that should respect opt-out
        advertising_cookie = CookieRecord(
            name="_fbp", 
            value="fb.1.123456789",
            domain="facebook.com",
            path="/",
            secure=False,
            http_only=False,
            first_party=False,
            essential=False,
            scenario_id="test"
        )
        
        cookies = [advertising_cookie]
        issues = self.engine.validate_cookie_policy(
            cookies, page_url, ComplianceFramework.CCPA, "test", "production"
        )
        
        # Should have CCPA-specific validation issues
        ccpa_issues = [issue for issue in issues 
                      if issue.compliance_framework == "CCPA"]
        assert len(ccpa_issues) >= 1
    
    def test_cookie_expiration_validation(self):
        """Test cookie expiration time validation."""
        page_url = "https://example.com"
        
        # Cookie with excessive expiration (> 2 years)
        long_lived_cookie = CookieRecord(
            name="persistent_cookie",
            value="value123",
            domain="example.com",
            path="/",
            expires=int((datetime.utcnow() + timedelta(days=800)).timestamp()),
            secure=True,
            http_only=False,
            first_party=True,
            scenario_id="test"
        )
        
        # Cookie with reasonable expiration
        normal_cookie = CookieRecord(
            name="normal_cookie",
            value="value456", 
            domain="example.com",
            path="/",
            expires=int((datetime.utcnow() + timedelta(days=30)).timestamp()),
            secure=True,
            http_only=False,
            first_party=True,
            scenario_id="test"
        )
        
        cookies = [long_lived_cookie, normal_cookie]
        issues = self.engine.validate_cookie_policy(
            cookies, page_url, ComplianceFramework.GDPR, "test", "production"
        )
        
        # Should flag excessive expiration
        expiration_issues = [issue for issue in issues if "expiration" in issue.message.lower()]
        assert len(expiration_issues) == 1
        assert expiration_issues[0].cookie_name == "persistent_cookie"
    
    def test_environment_specific_policies(self):
        """Test environment-specific policy enforcement."""
        page_url = "https://example.com"
        
        # Cookie that might be acceptable in dev but not prod
        debug_cookie = CookieRecord(
            name="debug_info", 
            value="debug_data",
            domain="example.com",
            path="/",
            secure=False,  # Might be OK in dev
            http_only=False,
            first_party=True,
            scenario_id="test"
        )
        
        cookies = [debug_cookie]
        
        # Test in development environment - more lenient
        dev_issues = self.engine.validate_cookie_policy(
            cookies, page_url, ComplianceFramework.GDPR, "test", "development"
        )
        
        # Test in production environment - stricter
        prod_issues = self.engine.validate_cookie_policy(
            cookies, page_url, ComplianceFramework.GDPR, "test", "production"
        )
        
        # Production should have more issues than development
        assert len(prod_issues) >= len(dev_issues)
    
    def test_policy_issue_severity_assignment(self):
        """Test proper severity assignment for policy issues."""
        page_url = "https://example.com"
        
        # High severity: session cookie without HttpOnly
        critical_cookie = CookieRecord(
            name="admin_session",
            value="admin123",
            domain="example.com",
            path="/admin",
            secure=False,  # High severity on HTTPS
            http_only=False,  # High severity for admin session
            first_party=True,
            essential=True,
            scenario_id="test"
        )
        
        # Medium severity: tracking cookie without proper attributes
        medium_cookie = CookieRecord(
            name="analytics",
            value="track123",
            domain="example.com",
            path="/", 
            secure=False,
            http_only=False,
            first_party=True,
            essential=False,
            scenario_id="test"
        )
        
        # Low severity: preference cookie with minor issue
        low_cookie = CookieRecord(
            name="theme",
            value="dark",
            domain="example.com",
            path="/",
            secure=True,
            http_only=False,
            same_site=None,  # Minor issue
            first_party=True,
            scenario_id="test"
        )
        
        cookies = [critical_cookie, medium_cookie, low_cookie]
        issues = self.engine.validate_cookie_policy(
            cookies, page_url, ComplianceFramework.GDPR, "test", "production"
        )
        
        # Check severity distribution
        high_issues = [issue for issue in issues if issue.severity == "high"]
        medium_issues = [issue for issue in issues if issue.severity == "medium"] 
        low_issues = [issue for issue in issues if issue.severity == "low"]
        
        assert len(high_issues) >= 1  # Admin session issues
        assert len(medium_issues) >= 1  # Analytics issues
        # Low issues might be present depending on policy strictness
    
    def test_bulk_policy_validation_performance(self):
        """Test policy validation performance with large cookie sets."""
        page_url = "https://example.com"
        
        # Generate large set of cookies
        cookies = []
        for i in range(500):
            cookies.append(CookieRecord(
                name=f"cookie_{i}",
                value=f"value_{i}",
                domain="example.com" if i % 2 == 0 else f"domain{i % 5}.com",
                path="/",
                secure=i % 3 == 0,
                http_only=i % 4 == 0,
                first_party=i % 2 == 0,
                scenario_id="test"
            ))
        
        # Validation should complete quickly
        import time
        start_time = time.time()
        issues = self.engine.validate_cookie_policy(
            cookies, page_url, ComplianceFramework.GDPR, "test", "production"
        )
        end_time = time.time()
        
        # Should complete in reasonable time
        assert (end_time - start_time) < 2.0  # Under 2 seconds
        assert isinstance(issues, list)
        
        # Should find multiple issues in the large set
        assert len(issues) > 0