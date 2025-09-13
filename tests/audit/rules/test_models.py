"""Unit tests for rule engine models and data structures."""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any

from app.audit.rules.models import (
    Rule, RuleResults, RuleSummary, Failure, Severity, RuleScope,
    CheckType, AppliesTo, CheckConfig, AlertConfig, AlertChannelType
)


class TestSeverity:
    """Test severity enumeration."""
    
    def test_severity_values(self):
        """Test severity enum values."""
        assert Severity.INFO == "info"
        assert Severity.WARNING == "warning" 
        assert Severity.CRITICAL == "critical"
    
    def test_severity_ordering(self):
        """Test severity comparison ordering."""
        # Test string comparison works correctly
        severities = [Severity.CRITICAL, Severity.INFO, Severity.WARNING]
        assert sorted(severities) == [Severity.CRITICAL, Severity.INFO, Severity.WARNING]


class TestFailure:
    """Test Failure model."""
    
    def test_failure_creation(self):
        """Test basic failure creation."""
        failure = Failure(
            check_id="test_check",
            severity=Severity.WARNING,
            message="Test failure message"
        )
        
        assert failure.check_id == "test_check"
        assert failure.severity == Severity.WARNING
        assert failure.message == "Test failure message"
        assert failure.details is None
        assert failure.evidence is None
        assert failure.context is None
    
    def test_failure_with_all_fields(self):
        """Test failure with all optional fields."""
        evidence = ["evidence1", "evidence2"]
        context = {"page_url": "https://example.com"}
        
        failure = Failure(
            check_id="detailed_check",
            severity=Severity.CRITICAL,
            message="Detailed failure",
            details="Additional details",
            evidence=evidence,
            context=context
        )
        
        assert failure.check_id == "detailed_check"
        assert failure.severity == Severity.CRITICAL
        assert failure.message == "Detailed failure"
        assert failure.details == "Additional details"
        assert failure.evidence == evidence
        assert failure.context == context
    
    def test_failure_serialization(self):
        """Test failure model serialization."""
        failure = Failure(
            check_id="serialize_test",
            severity=Severity.INFO,
            message="Serialization test",
            evidence=["item1", "item2"]
        )
        
        data = failure.model_dump()
        assert data["check_id"] == "serialize_test"
        assert data["severity"] == "info"
        assert data["message"] == "Serialization test"
        assert data["evidence"] == ["item1", "item2"]


class TestRuleSummary:
    """Test RuleSummary model."""
    
    def test_rule_summary_defaults(self):
        """Test rule summary with default values."""
        summary = RuleSummary()
        
        assert summary.total_rules == 0
        assert summary.passed_rules == 0
        assert summary.failed_rules == 0
        assert summary.total_failures == 0
        assert summary.critical_failures == 0
        assert summary.warning_failures == 0
        assert summary.info_failures == 0
        assert summary.execution_time_ms == 0.0
    
    def test_rule_summary_with_values(self):
        """Test rule summary with specific values."""
        summary = RuleSummary(
            total_rules=10,
            passed_rules=7,
            failed_rules=3,
            total_failures=5,
            critical_failures=1,
            warning_failures=2,
            info_failures=2,
            execution_time_ms=1500.0
        )
        
        assert summary.total_rules == 10
        assert summary.passed_rules == 7
        assert summary.failed_rules == 3
        assert summary.total_failures == 5
        assert summary.critical_failures == 1
        assert summary.warning_failures == 2
        assert summary.info_failures == 2
        assert summary.execution_time_ms == 1500.0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        summary = RuleSummary(total_rules=10, passed_rules=8)
        assert summary.success_rate == 80.0
        
        # Test zero total rules
        summary_empty = RuleSummary(total_rules=0, passed_rules=0)
        assert summary_empty.success_rate == 0.0
    
    def test_failure_rate_calculation(self):
        """Test failure rate calculation."""
        summary = RuleSummary(total_rules=10, failed_rules=3)
        assert summary.failure_rate == 30.0
        
        # Test zero total rules
        summary_empty = RuleSummary(total_rules=0, failed_rules=0)
        assert summary_empty.failure_rate == 0.0


class TestAppliesTo:
    """Test AppliesTo model."""
    
    def test_applies_to_creation(self):
        """Test AppliesTo model creation."""
        applies_to = AppliesTo(
            urls=["https://example.com/*"],
            environments=["production"],
            scope=RuleScope.PAGE
        )
        
        assert applies_to.urls == ["https://example.com/*"]
        assert applies_to.environments == ["production"]
        assert applies_to.scope == RuleScope.PAGE
    
    def test_applies_to_defaults(self):
        """Test AppliesTo with default values."""
        applies_to = AppliesTo()
        
        assert applies_to.urls == []
        assert applies_to.environments == []
        assert applies_to.scope == RuleScope.PAGE


class TestCheckConfig:
    """Test CheckConfig model."""
    
    def test_check_config_basic(self):
        """Test basic CheckConfig creation."""
        config = CheckConfig(
            type=CheckType.PRESENCE,
            parameters={"url_pattern": "analytics.js"}
        )
        
        assert config.type == CheckType.PRESENCE
        assert config.parameters == {"url_pattern": "analytics.js"}
        assert config.timeout_seconds == 30
        assert config.retry_count == 0
    
    def test_check_config_with_all_options(self):
        """Test CheckConfig with all options."""
        config = CheckConfig(
            type=CheckType.DUPLICATE,
            parameters={"window_seconds": 60},
            timeout_seconds=45,
            retry_count=2,
            enabled=False
        )
        
        assert config.type == CheckType.DUPLICATE
        assert config.parameters == {"window_seconds": 60}
        assert config.timeout_seconds == 45
        assert config.retry_count == 2
        assert config.enabled is False


class TestRule:
    """Test Rule model."""
    
    def test_rule_creation(self):
        """Test basic rule creation."""
        check_config = CheckConfig(
            type=CheckType.PRESENCE,
            parameters={"url_pattern": "gtm.js"}
        )
        
        rule = Rule(
            id="test_rule",
            name="Test GTM Rule",
            description="Test GTM loading",
            severity=Severity.CRITICAL,
            check=check_config
        )
        
        assert rule.id == "test_rule"
        assert rule.name == "Test GTM Rule"
        assert rule.description == "Test GTM loading"
        assert rule.severity == Severity.CRITICAL
        assert rule.check == check_config
        assert rule.applies_to is not None
        assert rule.enabled is True
    
    def test_rule_with_applies_to(self):
        """Test rule with custom AppliesTo."""
        applies_to = AppliesTo(
            urls=["https://shop.example.com/*"],
            environments=["production", "staging"]
        )
        
        check_config = CheckConfig(
            type=CheckType.ABSENCE,
            parameters={"url_pattern": "test-analytics.js"}
        )
        
        rule = Rule(
            id="prod_rule",
            name="Production Rule",
            description="No test analytics in production",
            severity=Severity.WARNING,
            check=check_config,
            applies_to=applies_to,
            enabled=False
        )
        
        assert rule.id == "prod_rule"
        assert rule.applies_to.urls == ["https://shop.example.com/*"]
        assert rule.applies_to.environments == ["production", "staging"]
        assert rule.enabled is False


class TestRuleResults:
    """Test RuleResults model."""
    
    def test_rule_results_creation(self):
        """Test RuleResults creation."""
        failures = [
            Failure(
                check_id="test1",
                severity=Severity.CRITICAL,
                message="Critical failure"
            ),
            Failure(
                check_id="test2", 
                severity=Severity.WARNING,
                message="Warning failure"
            )
        ]
        
        summary = RuleSummary(
            total_rules=5,
            passed_rules=3,
            failed_rules=2,
            total_failures=2,
            critical_failures=1,
            warning_failures=1
        )
        
        results = RuleResults(
            failures=failures,
            summary=summary
        )
        
        assert len(results.failures) == 2
        assert results.summary.total_rules == 5
        assert results.summary.failed_rules == 2
        assert results.context_info == {}
        assert isinstance(results.evaluation_time, datetime)
    
    def test_rule_results_with_context(self):
        """Test RuleResults with context info."""
        context_info = {
            "environment": "production",
            "urls": ["https://example.com"],
            "evaluator_version": "1.0"
        }
        
        results = RuleResults(
            failures=[],
            summary=RuleSummary(),
            context_info=context_info
        )
        
        assert results.context_info == context_info


class TestAlertConfig:
    """Test AlertConfig model."""
    
    def test_alert_config_creation(self):
        """Test basic AlertConfig creation."""
        config = AlertConfig(
            enabled=True,
            channels=[AlertChannelType.WEBHOOK, AlertChannelType.EMAIL]
        )
        
        assert config.enabled is True
        assert config.channels == [AlertChannelType.WEBHOOK, AlertChannelType.EMAIL]
        assert config.webhook_url is None
        assert config.email_recipients == []
        assert config.template == {}
    
    def test_alert_config_full(self):
        """Test AlertConfig with all fields."""
        template = {
            "title": "Alert: {{rule_name}}",
            "message": "Rule failed: {{failure_message}}"
        }
        
        config = AlertConfig(
            enabled=True,
            channels=[AlertChannelType.EMAIL],
            email_recipients=["admin@example.com", "dev@example.com"],
            template=template
        )
        
        assert config.enabled is True
        assert config.channels == [AlertChannelType.EMAIL]
        assert config.email_recipients == ["admin@example.com", "dev@example.com"]
        assert config.template == template


class TestModelValidation:
    """Test model validation and edge cases."""
    
    def test_invalid_severity(self):
        """Test handling of invalid severity values."""
        with pytest.raises(ValueError):
            Failure(
                check_id="test",
                severity="invalid",  # type: ignore
                message="test"
            )
    
    def test_empty_check_id(self):
        """Test validation of empty check_id."""
        failure = Failure(
            check_id="",
            severity=Severity.INFO,
            message="Empty check ID test"
        )
        # Should not raise an error - empty string is valid
        assert failure.check_id == ""
    
    def test_none_evidence_handling(self):
        """Test handling of None evidence."""
        failure = Failure(
            check_id="test",
            severity=Severity.INFO,
            message="test",
            evidence=None
        )
        assert failure.evidence is None
    
    def test_rule_id_uniqueness(self):
        """Test that rule IDs can be duplicated (validation is external)."""
        rule1 = Rule(
            id="duplicate_id",
            name="Rule 1",
            description="First rule",
            severity=Severity.INFO,
            check=CheckConfig(type=CheckType.PRESENCE, parameters={})
        )
        
        rule2 = Rule(
            id="duplicate_id", 
            name="Rule 2",
            description="Second rule",
            severity=Severity.WARNING,
            check=CheckConfig(type=CheckType.ABSENCE, parameters={})
        )
        
        # Should not raise error - uniqueness is enforced at higher level
        assert rule1.id == rule2.id
    
    def test_model_serialization_roundtrip(self):
        """Test that models can be serialized and deserialized."""
        original_rule = Rule(
            id="roundtrip_test",
            name="Roundtrip Test",
            description="Testing serialization roundtrip",
            severity=Severity.WARNING,
            check=CheckConfig(
                type=CheckType.PRESENCE,
                parameters={"url_pattern": "test.js", "timeout": 5000}
            ),
            applies_to=AppliesTo(
                urls=["https://*.example.com/*"],
                environments=["test"]
            )
        )
        
        # Serialize to dict
        data = original_rule.model_dump()
        
        # Deserialize back to model
        restored_rule = Rule(**data)
        
        assert restored_rule.id == original_rule.id
        assert restored_rule.name == original_rule.name
        assert restored_rule.check.type == original_rule.check.type
        assert restored_rule.check.parameters == original_rule.check.parameters
        assert restored_rule.applies_to.urls == original_rule.applies_to.urls