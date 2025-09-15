"""Unit tests for cookie classification engine."""

import pytest
from unittest.mock import Mock, patch
from urllib.parse import urlparse

from app.audit.cookies.classification import CookieClassifier, CookieCategory
from app.audit.cookies.models import CookieRecord
from app.audit.cookies.config import PrivacyConfiguration


class TestCookieClassifier:
    """Test CookieClassifier functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PrivacyConfiguration()
        self.classifier = CookieClassifier(self.config)
    
    def test_etld_plus_one_extraction(self):
        """Test eTLD+1 domain extraction."""
        # Test basic domains
        assert self.classifier._extract_etld_plus_one("example.com") == "example.com"
        assert self.classifier._extract_etld_plus_one("www.example.com") == "example.com"
        assert self.classifier._extract_etld_plus_one("api.example.com") == "example.com"
        
        # Test multi-level subdomains
        assert self.classifier._extract_etld_plus_one("cdn.assets.example.com") == "example.com"
        
        # Test different TLDs
        assert self.classifier._extract_etld_plus_one("example.co.uk") == "example.co.uk"
        assert self.classifier._extract_etld_plus_one("www.example.co.uk") == "example.co.uk"
        
        # Test edge cases
        assert self.classifier._extract_etld_plus_one("localhost") == "localhost"
        assert self.classifier._extract_etld_plus_one("192.168.1.1") == "192.168.1.1"
    
    def test_first_party_classification(self):
        """Test first-party vs third-party classification."""
        page_url = "https://example.com/page"
        
        # First-party cookies
        first_party_cookie = CookieRecord(
            name="session_id",
            value="abc123",
            domain="example.com",
            path="/",
            secure=True,
            http_only=True,
            first_party=True,  # This will be overridden by classification
            scenario_id="test"
        )
        
        subdomain_cookie = CookieRecord(
            name="api_token", 
            value="token123",
            domain="api.example.com",
            path="/",
            secure=True,
            http_only=False,
            first_party=True,
            scenario_id="test"
        )
        
        # Third-party cookies
        third_party_cookie = CookieRecord(
            name="tracking_id",
            value="xyz789",
            domain="ads.googletagmanager.com",
            path="/",
            secure=False,
            http_only=False,
            first_party=False,
            scenario_id="test"
        )
        
        cookies = [first_party_cookie, subdomain_cookie, third_party_cookie]
        classified = self.classifier.classify_cookies(cookies, page_url)
        
        # Check first-party classification
        session_cookie = next(c for c in classified if c.name == "session_id")
        api_cookie = next(c for c in classified if c.name == "api_token")
        tracking_cookie = next(c for c in classified if c.name == "tracking_id")
        
        assert session_cookie.first_party is True
        assert api_cookie.first_party is True  # Same eTLD+1
        assert tracking_cookie.first_party is False
    
    def test_essential_cookie_detection(self):
        """Test essential cookie detection heuristics."""
        page_url = "https://example.com"
        
        # Essential cookie patterns
        session_cookie = CookieRecord(
            name="PHPSESSID",
            value="abc123",
            domain="example.com",
            path="/",
            secure=True,
            http_only=True,
            first_party=True,
            scenario_id="test"
        )
        
        csrf_cookie = CookieRecord(
            name="csrf_token",
            value="token123",
            domain="example.com", 
            path="/",
            secure=True,
            http_only=True,
            first_party=True,
            scenario_id="test"
        )
        
        auth_cookie = CookieRecord(
            name="auth_session",
            value="session123",
            domain="example.com",
            path="/",
            secure=True,
            http_only=True,
            first_party=True,
            scenario_id="test"
        )
        
        # Non-essential cookie
        analytics_cookie = CookieRecord(
            name="_ga",
            value="GA1.2.123456789",
            domain="example.com",
            path="/",
            secure=False,
            http_only=False,
            first_party=True,
            scenario_id="test"
        )
        
        cookies = [session_cookie, csrf_cookie, auth_cookie, analytics_cookie]
        classified = self.classifier.classify_cookies(cookies, page_url)
        
        # Check essential classification
        session = next(c for c in classified if c.name == "PHPSESSID")
        csrf = next(c for c in classified if c.name == "csrf_token")
        auth = next(c for c in classified if c.name == "auth_session")
        analytics = next(c for c in classified if c.name == "_ga")
        
        assert session.essential is True
        assert csrf.essential is True  
        assert auth.essential is True
        assert analytics.essential is False
    
    def test_cookie_categorization(self):
        """Test cookie categorization by purpose."""
        page_url = "https://example.com"
        
        # Different cookie types
        cookies = [
            # Functional
            CookieRecord(
                name="language_pref",
                value="en-US", 
                domain="example.com",
                path="/",
                secure=True,
                http_only=False,
                first_party=True,
                scenario_id="test"
            ),
            # Analytics
            CookieRecord(
                name="_ga",
                value="GA1.2.123456789",
                domain="example.com",
                path="/",
                secure=False,
                http_only=False,
                first_party=True,
                scenario_id="test"
            ),
            # Marketing
            CookieRecord(
                name="_fbp",
                value="fb.1.123456789", 
                domain="example.com",
                path="/",
                secure=False,
                http_only=False,
                first_party=True,
                scenario_id="test"
            ),
            # Unknown third-party
            CookieRecord(
                name="unknown_cookie",
                value="unknown_value",
                domain="thirdparty.com",
                path="/",
                secure=False,
                http_only=False,
                first_party=False,
                scenario_id="test"
            )
        ]
        
        classified = self.classifier.classify_cookies(cookies, page_url)
        
        # Check categories
        lang_cookie = next(c for c in classified if c.name == "language_pref")
        ga_cookie = next(c for c in classified if c.name == "_ga")
        fb_cookie = next(c for c in classified if c.name == "_fbp")
        unknown_cookie = next(c for c in classified if c.name == "unknown_cookie")
        
        assert lang_cookie.category == CookieCategory.FUNCTIONAL
        assert ga_cookie.category == CookieCategory.ANALYTICS
        assert fb_cookie.category == CookieCategory.MARKETING
        assert unknown_cookie.category == CookieCategory.UNKNOWN
    
    def test_subdomain_edge_cases(self):
        """Test subdomain classification edge cases."""
        page_url = "https://www.example.com"
        
        # Test various subdomain patterns
        cookies = [
            # Bare domain 
            CookieRecord(
                name="cookie1",
                value="value1",
                domain="example.com",
                path="/",
                secure=True,
                http_only=False,
                first_party=True,
                scenario_id="test"
            ),
            # www subdomain
            CookieRecord(
                name="cookie2", 
                value="value2",
                domain="www.example.com",
                path="/",
                secure=True,
                http_only=False,
                first_party=True,
                scenario_id="test"
            ),
            # API subdomain
            CookieRecord(
                name="cookie3",
                value="value3",
                domain="api.example.com", 
                path="/",
                secure=True,
                http_only=False,
                first_party=True,
                scenario_id="test"
            ),
            # CDN subdomain
            CookieRecord(
                name="cookie4",
                value="value4",
                domain="cdn.example.com",
                path="/",
                secure=True,
                http_only=False,
                first_party=True,
                scenario_id="test"
            )
        ]
        
        classified = self.classifier.classify_cookies(cookies, page_url)
        
        # All should be classified as first-party
        for cookie in classified:
            assert cookie.first_party is True, f"Cookie {cookie.name} should be first-party"
    
    def test_classification_with_custom_rules(self):
        """Test classification with custom configuration rules."""
        # Create config with custom rules using ClassificationConfig
        from app.audit.cookies.config import ClassificationConfig
        
        config = PrivacyConfiguration()
        config.classification = ClassificationConfig(
            essential_patterns=[
                r"^custom_session_",
                r"_auth$"
            ],
            non_essential_patterns=[
                r"^analytics_",
                r"^marketing_"
            ]
        )
        
        classifier = CookieClassifier(config)
        page_url = "https://example.com"
        
        cookies = [
            CookieRecord(
                name="custom_session_abc123", 
                value="value",
                domain="example.com",
                path="/",
                secure=True,
                http_only=True,
                first_party=True,
                scenario_id="test"
            ),
            CookieRecord(
                name="user_auth",
                value="value",
                domain="example.com",
                path="/",
                secure=True,
                http_only=True, 
                first_party=True,
                scenario_id="test"
            ),
            CookieRecord(
                name="analytics_tracking",
                value="value",
                domain="example.com",
                path="/",
                secure=False,
                http_only=False,
                first_party=True,
                scenario_id="test"
            )
        ]
        
        classified = classifier.classify_cookies(cookies, page_url)
        
        session_cookie = next(c for c in classified if c.name == "custom_session_abc123")
        auth_cookie = next(c for c in classified if c.name == "user_auth") 
        analytics_cookie = next(c for c in classified if c.name == "analytics_tracking")
        
        assert session_cookie.essential is True
        assert auth_cookie.essential is True
        assert analytics_cookie.essential is False
    
    def test_performance_with_large_cookie_set(self):
        """Test classification performance with large number of cookies."""
        page_url = "https://example.com"
        
        # Generate large set of cookies
        cookies = []
        for i in range(1000):
            domain = f"domain{i % 10}.com" if i % 3 == 0 else "example.com"
            cookies.append(CookieRecord(
                name=f"cookie_{i}",
                value=f"value_{i}",
                domain=domain,
                path="/",
                secure=i % 2 == 0,
                http_only=i % 3 == 0,
                first_party=domain == "example.com",
                scenario_id="test"
            ))
        
        # Classification should complete quickly
        import time
        start_time = time.time()
        classified = self.classifier.classify_cookies(cookies, page_url)
        end_time = time.time()
        
        assert len(classified) == 1000
        assert (end_time - start_time) < 1.0  # Should complete in under 1 second
        
        # Verify first-party classification is correct
        first_party_count = sum(1 for c in classified if c.first_party)
        third_party_count = len(classified) - first_party_count
        
        # About 2/3 should be first-party based on our generation logic
        assert first_party_count > 600
        assert third_party_count > 300