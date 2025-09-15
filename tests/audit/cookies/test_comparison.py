"""Unit tests for cross-scenario comparison engine."""

import pytest
from typing import List

from app.audit.cookies.comparison import ScenarioComparator, CookieChangeType
from app.audit.cookies.models import CookieRecord, CookieDiff, ScenarioCookieReport
from app.audit.cookies.config import PrivacyConfiguration


class TestScenarioComparator:
    """Test ScenarioComparator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PrivacyConfiguration()
        self.comparator = ScenarioComparator()
    
    def test_basic_cookie_diff(self):
        """Test basic cookie difference detection.""" 
        # Baseline cookies
        baseline_cookies = [
            CookieRecord(
                name="session_id",
                value="session123",
                domain="example.com",
                path="/",
                size=20,
                secure=True,
                http_only=True,
                is_first_party=True
            ),
            CookieRecord(
                name="tracking_id",
                value="track123",
                domain="analytics.com",
                path="/",
                size=15,
                secure=False,
                http_only=False,
                is_first_party=False
            )
        ]
        
        # GPC cookies (tracking cookie removed)
        gpc_cookies = [
            CookieRecord(
                name="session_id",
                value="session123",
                domain="example.com",
                path="/",
                size=20,
                secure=True,
                http_only=True,
                is_first_party=True,
                            ),
            CookieRecord(
                name="privacy_signal",
                value="gpc_detected",
                domain="example.com",
                path="/",
                secure=True,
                http_only=False,
                is_first_party=True,
                            )
        ]
        
        # Compare scenarios
        diff = self.comparator.compare_cookies(
            baseline_cookies, gpc_cookies, "baseline", "gpc_on"
        )
        
        # Verify differences
        assert diff.scenario_a_id == "baseline"
        assert diff.scenario_b_id == "gpc_on"
        
        # Should have one added cookie (privacy_signal)
        assert len(diff.added_cookies) == 1
        assert diff.added_cookies[0].name == "privacy_signal"
        
        # Should have one removed cookie (tracking_id)  
        assert len(diff.removed_cookies) == 1
        assert diff.removed_cookies[0].name == "tracking_id"
        
        # Should have one unchanged cookie (session_id)
        assert len(diff.unchanged_cookies) == 1
        assert diff.unchanged_cookies[0].name == "session_id"
        
        # No modified cookies in this case
        assert len(diff.modified_cookies) == 0
    
    def test_cookie_attribute_changes(self):
        """Test detection of cookie attribute changes."""
        # Baseline cookies
        baseline_cookies = [
            CookieRecord(
                name="user_pref",
                value="theme=dark",
                domain="example.com", 
                path="/",
                expires=1735689600,
                secure=False,
                http_only=False,
                same_site="Lax",
                is_first_party=True,
            )
        ]
        
        # Modified cookies (same cookie but different attributes)
        modified_cookies = [
            CookieRecord(
                name="user_pref",
                value="theme=light",  # Different value
                domain="example.com",
                path="/",
                expires=1735689600,
                secure=True,  # Different secure flag
                http_only=False,
                same_site="Strict",  # Different SameSite
                is_first_party=True,
                            )
        ]
        
        # Compare
        diff = self.comparator.compare_cookies(
            baseline_cookies, modified_cookies, "baseline", "modified"
        )
        
        # Should detect modification
        assert len(diff.modified_cookies) == 1
        assert len(diff.added_cookies) == 0
        assert len(diff.removed_cookies) == 0
        assert len(diff.unchanged_cookies) == 0
        
        modified_cookie = diff.modified_cookies[0]
        assert modified_cookie.cookie_name == "user_pref"

        # Check specific changes
        changes = modified_cookie.attribute_changes
        assert "value" in changes
        assert "secure" in changes
        assert "same_site" in changes
        
        assert changes["value"]["from"] == "theme=dark"
        assert changes["value"]["to"] == "theme=light"
        assert changes["secure"]["from"] is False
        assert changes["secure"]["to"] is True
    
    def test_complex_scenario_comparison(self):
        """Test complex scenario with multiple types of changes."""
        # Baseline scenario (many cookies)
        baseline_cookies = [
            CookieRecord(
                name="essential_session", 
                value="sess1",
                domain="example.com",
                path="/",
                secure=True,
                http_only=True,
                is_first_party=True,
            ),
            CookieRecord(
                name="google_analytics",
                value="GA1.2.123",
                domain="google-analytics.com", 
                path="/",
                secure=False,
                http_only=False,
                is_first_party=False,
            ),
            CookieRecord(
                name="facebook_pixel",
                value="fb_pixel_123",
                domain="facebook.com",
                path="/",
                secure=False,
                http_only=False,
                is_first_party=False,
            ),
            CookieRecord(
                name="user_preferences",
                value="lang=en",
                domain="example.com",
                path="/",
                secure=False,
                http_only=False,
                is_first_party=True,
            )
        ]
        
        # GPC scenario (tracking cookies removed, preferences modified)
        gpc_cookies = [
            CookieRecord(
                name="essential_session",
                value="sess1", 
                domain="example.com",
                path="/",
                secure=True,
                http_only=True,
                is_first_party=True,
                            ),
            CookieRecord(
                name="user_preferences",
                value="lang=en,privacy=gpc",  # Modified
                domain="example.com",
                path="/",
                secure=True,  # Enhanced security
                http_only=False,
                is_first_party=True,
                            ),
            CookieRecord(
                name="gpc_signal",  # New cookie
                value="enabled",
                domain="example.com",
                path="/",
                secure=True,
                http_only=False,
                is_first_party=True,
                            )
        ]
        
        # Compare
        diff = self.comparator.compare_cookies(
            baseline_cookies, gpc_cookies, "baseline", "gpc_on"
        )
        
        # Verify complex changes
        assert len(diff.unchanged_cookies) == 1  # essential_session
        assert len(diff.modified_cookies) == 1   # user_preferences
        assert len(diff.added_cookies) == 1      # gpc_signal
        assert len(diff.removed_cookies) == 2    # google_analytics, facebook_pixel
        
        # Check specific changes
        assert diff.unchanged_cookies[0].name == "essential_session"
        assert diff.added_cookies[0].name == "gpc_signal"
        assert diff.modified_cookies[0].cookie_name == "user_preferences"
        
        removed_names = [c.name for c in diff.removed_cookies]
        assert "google_analytics" in removed_names
        assert "facebook_pixel" in removed_names
    
    def test_compare_scenario_reports(self):
        """Test comparison of full scenario reports."""
        # Create baseline report
        baseline_report = ScenarioCookieReport(
            scenario_id="baseline",
            scenario_name="Baseline Test",
            page_url="https://example.com",
            cookies=[
                CookieRecord(
                    name="tracking_cookie",
                    value="track123",
                    domain="tracker.com",
                    path="/",
                    secure=False,
                    http_only=False,
                    is_first_party=False,
                    )
            ],
            policy_issues=[],
            errors=[]
        )
        
        # Create GPC report (no tracking cookies)
        gpc_report = ScenarioCookieReport(
            scenario_id="gpc_on",
            scenario_name="GPC Enabled",
            page_url="https://example.com", 
            cookies=[],  # Tracking blocked
            policy_issues=[],
            errors=[]
        )
        
        # Compare scenarios using existing method
        diff = self.comparator.compare_scenarios(baseline_report, gpc_report)

        # Verify comparison result
        assert diff.baseline_scenario == "baseline"
        assert diff.variant_scenario == "gpc_on"
        assert len(baseline_report.cookies) == 1  # Baseline had tracking cookie
        assert len(gpc_report.cookies) == 0       # GPC blocked tracking
        
        # Check diff details
        assert len(diff.removed_cookies) == 1
        assert diff.removed_cookies[0].name == "tracking_cookie"
    
    def test_privacy_impact_analysis(self):
        """Test privacy impact analysis between scenarios."""
        # Baseline with tracking cookies
        baseline_cookies = [
            CookieRecord(
                name="essential_session",
                value="sess1",
                domain="example.com",
                path="/",
                secure=True,
                http_only=True,
                is_first_party=True,
                essential=True,
            ),
            CookieRecord(
                name="google_ads",
                value="ads123",
                domain="googleadservices.com",
                path="/",
                secure=False,
                http_only=False,
                is_first_party=False,
                essential=False,
            ),
            CookieRecord(
                name="facebook_tracking",
                value="fb123",
                domain="facebook.com",
                path="/",
                secure=False,
                http_only=False,
                is_first_party=False,
                essential=False,
            )
        ]
        
        # Privacy-respecting scenario
        privacy_cookies = [
            CookieRecord(
                name="essential_session",
                value="sess1",
                domain="example.com",
                path="/",
                secure=True,
                http_only=True,
                is_first_party=True,
                essential=True,
                            )
        ]
        
        # Analyze privacy impact
        scenario_reports = {
            "baseline": {"cookies": baseline_cookies},
            "privacy": {"cookies": privacy_cookies}
        }
        impact = self.comparator.analyze_privacy_impact(scenario_reports)
        
        # Verify impact analysis
        assert impact["tracking_cookies_blocked"] == 2
        assert impact["essential_cookies_preserved"] == 1
        assert impact["third_party_cookies_reduced"] == 2
        assert impact["privacy_improvement_score"] > 50  # Significant improvement
        
        # Check detailed breakdown
        assert len(impact["blocked_trackers"]) == 2
        blocked_domains = [t["domain"] for t in impact["blocked_trackers"]]
        assert "googleadservices.com" in blocked_domains
        assert "facebook.com" in blocked_domains
    
    def test_statistical_analysis(self):
        """Test statistical analysis of cookie differences."""
        # Generate multiple comparison data points  
        comparisons = []
        
        for i in range(10):
            baseline = [
                CookieRecord(
                    name=f"cookie_{j}",
                    value=f"value_{j}",
                    domain="example.com" if j < 5 else "tracker.com",
                    path="/",
                    secure=j % 2 == 0,
                    http_only=j % 3 == 0,
                    is_first_party=j < 5,
                    )
                for j in range(10)
            ]
            
            gpc = [
                CookieRecord(
                    name=f"cookie_{j}",
                    value=f"value_{j}",
                    domain="example.com",
                    path="/",
                    secure=j % 2 == 0,
                    http_only=j % 3 == 0,
                    is_first_party=True
                )
                for j in range(5)  # Only first-party cookies remain
            ]
            
            diff = self.comparator.compare_cookies(
                baseline, gpc, "baseline", "gpc"
            )
            comparisons.append(diff)
        
        # Analyze statistics
        stats = self.comparator.generate_comparison_statistics(comparisons)
        
        # Verify statistics
        assert stats["total_comparisons"] == 10
        assert stats["avg_cookies_removed"] == 5.0  # Consistent removal
        assert stats["avg_cookies_added"] == 0.0    # No additions
        assert "removal_rate" in stats
        assert stats["removal_rate"] > 0
    
    def test_diff_visualization_data(self):
        """Test generation of data for diff visualization."""
        # Create simple diff
        baseline_cookies = [
            CookieRecord(name="keep", value="v", domain="example.com", path="/", 
                        secure=True, http_only=False, is_first_party=True),
            CookieRecord(name="remove", value="v", domain="tracker.com", path="/",
                        secure=False, http_only=False, is_first_party=False)
        ]
        
        gpc_cookies = [
            CookieRecord(name="keep", value="v", domain="example.com", path="/",
                        secure=True, http_only=False, is_first_party=True),
            CookieRecord(name="add", value="v", domain="example.com", path="/",
                        secure=True, http_only=False, is_first_party=True)
        ]
        
        diff = self.comparator.compare_cookies(
            baseline_cookies, gpc_cookies, "baseline", "gpc"
        )
        
        # Generate visualization data
        viz_data = self.comparator.generate_visualization_data(diff)
        
        # Verify visualization data structure
        assert "summary" in viz_data
        assert "details" in viz_data
        assert "timeline" in viz_data
        
        summary = viz_data["summary"]
        assert summary["added"] == 1
        assert summary["removed"] == 1
        assert summary["unchanged"] == 1
        assert summary["modified"] == 0
        
        # Check details for UI consumption
        details = viz_data["details"]
        assert len(details["added"]) == 1
        assert len(details["removed"]) == 1
        assert details["added"][0]["name"] == "add"
        assert details["removed"][0]["name"] == "remove"
    
    def test_performance_with_large_cookie_sets(self):
        """Test comparison performance with large cookie sets."""
        # Generate large cookie sets
        baseline_size = 1000
        gpc_size = 500
        
        baseline_cookies = [
            CookieRecord(
                name=f"cookie_{i}",
                value=f"value_{i}",
                domain=f"domain{i % 10}.com",
                path="/",
                secure=i % 2 == 0,
                http_only=i % 3 == 0,
                is_first_party=i % 2 == 0,
            )
            for i in range(baseline_size)
        ]
        
        gpc_cookies = [
            CookieRecord(
                name=f"cookie_{i}",
                value=f"value_{i}",
                domain=f"domain{i % 10}.com", 
                path="/",
                secure=i % 2 == 0,
                http_only=i % 3 == 0,
                is_first_party=i % 2 == 0
            )
            for i in range(gpc_size)
        ]
        
        # Time the comparison
        import time
        start_time = time.time()
        diff = self.comparator.compare_cookies(
            baseline_cookies, gpc_cookies, "baseline", "gpc"
        )
        end_time = time.time()
        
        # Verify performance and correctness
        assert (end_time - start_time) < 1.0  # Should complete in under 1 second
        assert len(diff.removed_cookies) == 500  # 1000 - 500
        assert len(diff.unchanged_cookies) == 500  # Overlapping cookies
        assert len(diff.added_cookies) == 0
    
    def test_cookie_matching_logic(self):
        """Test cookie matching logic for comparisons."""
        # Test exact matches
        cookie1 = CookieRecord(
            name="test", value="v1", domain="example.com", path="/",
            secure=True, http_only=False, is_first_party=True
        )
        cookie2 = CookieRecord(
            name="test", value="v2", domain="example.com", path="/", 
            secure=False, http_only=True, is_first_party=True
        )
        
        # Should match by (name, domain, path)
        assert self.comparator._cookies_match(cookie1, cookie2) is True
        
        # Test non-matches
        cookie3 = CookieRecord(
            name="different", value="v1", domain="example.com", path="/",
            secure=True, http_only=False, is_first_party=True        )
        
        assert self.comparator._cookies_match(cookie1, cookie3) is False
        
        # Different domain
        cookie4 = CookieRecord(
            name="test", value="v1", domain="other.com", path="/",
            secure=True, http_only=False, is_first_party=True        )
        
        assert self.comparator._cookies_match(cookie1, cookie4) is False