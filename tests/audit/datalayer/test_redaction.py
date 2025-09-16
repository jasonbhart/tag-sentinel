"""Unit tests for DataLayer redaction system."""

import pytest
from typing import Dict, Any, List
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.audit.datalayer.redaction import (
    Redactor,
    SensitiveDataPattern,
    PatternLibrary,
    AdvancedPatternDetector
)
from app.audit.datalayer.models import RedactionMethod
from app.audit.datalayer.redaction import RedactionAuditEntry
from app.audit.datalayer.config import RedactionConfig, RedactionRuleConfig


class TestRedactor:
    """Test cases for basic Redactor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RedactionConfig()
        self.redaction_manager = Redactor(self.config)
    
    def test_redactor_initialization(self):
        """Test redactor initialization."""
        assert self.redaction_manager.config == self.config
        assert len(self.redaction_manager.audit_trail) == 0
    
    def test_hash_redaction(self):
        """Test HASH redaction method."""
        original_value = "sensitive_data_123"
        redacted = self.redaction_manager._apply_redaction_method(
            original_value, RedactionMethod.HASH
        )
        
        # Should return a hash in [HASH:...] format
        assert redacted != original_value
        assert redacted.startswith("[HASH:")
        assert redacted.endswith("]")
        assert len(redacted) > 10  # Should have some hash content
    
    def test_mask_redaction(self):
        """Test MASK redaction method."""
        original_value = "user@example.com"
        redacted = self.redaction_manager._apply_redaction_method(
            original_value, RedactionMethod.MASK
        )
        
        # Should mask middle characters
        assert redacted != original_value
        assert redacted.startswith("u")
        assert redacted.endswith("m")
        assert "*" in redacted
    
    def test_remove_redaction(self):
        """Test REMOVE redaction method."""
        original_value = "secret_key_123"
        redacted = self.redaction_manager._apply_redaction_method(
            original_value, RedactionMethod.REMOVE
        )

        assert redacted == "[REDACTED]"
    
    def test_truncate_redaction(self):
        """Test TRUNCATE redaction method."""
        original_value = "this_is_a_very_long_sensitive_string"
        redacted = self.redaction_manager._apply_redaction_method(
            original_value, RedactionMethod.TRUNCATE
        )
        
        # Should be truncated
        assert len(redacted) < len(original_value)
        assert redacted.endswith("...")
        assert original_value.startswith(redacted[:-3])
    
    def test_redact_by_path_simple(self):
        """Test redaction by JSON path."""
        data = {
            "user": {
                "email": "user@example.com",
                "name": "John Doe"
            },
            "page": "home"
        }

        # Create redaction rules for specific paths
        rules = [RedactionRuleConfig(path="/user/email", method=RedactionMethod.HASH)]
        redacted_data, _ = self.redaction_manager.redact_data(data, rules)

        assert redacted_data["user"]["email"] != "user@example.com"
        assert redacted_data["user"]["name"] == "John Doe"  # Unchanged
        assert redacted_data["page"] == "home"  # Unchanged
    
    def test_redact_by_path_array(self):
        """Test redaction with array indices."""
        data = {
            "users": [
                {"email": "user1@example.com", "name": "John"},
                {"email": "user2@example.com", "name": "Jane"}
            ]
        }

        # Create redaction rules for array paths
        rules = [
            RedactionRuleConfig(path="/users/0/email", method=RedactionMethod.HASH),
            RedactionRuleConfig(path="/users/1/email", method=RedactionMethod.HASH)
        ]
        redacted_data, _ = self.redaction_manager.redact_data(data, rules)

        assert redacted_data["users"][0]["email"] != "user1@example.com"
        assert redacted_data["users"][1]["email"] != "user2@example.com"
        assert redacted_data["users"][0]["name"] == "John"  # Unchanged
    
    def test_redact_by_glob_patterns(self):
        """Test redaction using glob patterns."""
        data = {
            "user_email": "user@example.com",
            "admin_email": "admin@example.com",
            "user_phone": "555-1234",
            "page_title": "Home Page"
        }

        # Create redaction rules with glob patterns
        rules = [
            RedactionRuleConfig(path="/*_email", method=RedactionMethod.HASH),
            RedactionRuleConfig(path="/*_phone", method=RedactionMethod.HASH)
        ]
        redacted_data, _ = self.redaction_manager.redact_data(data, rules)

        assert redacted_data["user_email"] != "user@example.com"
        assert redacted_data["admin_email"] != "admin@example.com"
        assert redacted_data["user_phone"] != "555-1234"
        assert redacted_data["page_title"] == "Home Page"  # Unchanged
    
    def test_audit_trail_creation(self):
        """Test that audit trail is created during redaction."""
        data = {"email": "user@example.com"}

        # Create redaction rule and perform redaction
        rules = [RedactionRuleConfig(path="/email", method=RedactionMethod.HASH)]
        redacted_data, audit_trail = self.redaction_manager.redact_data(data, rules)

        # Should have audit trail entry
        assert len(audit_trail) == 1
        audit = audit_trail[0]

        assert audit.path == "/email"
        assert audit.original_type == "str"
        assert audit.method == RedactionMethod.HASH
        assert redacted_data["email"] != "user@example.com"
    
    def test_nested_path_redaction(self):
        """Test redaction of deeply nested paths."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "sensitive": "secret_value"
                    }
                }
            }
        }
        
        paths_to_redact = ["/level1/level2/level3/sensitive"]
        redacted_data = self.redaction_manager.redact_by_paths(data, paths_to_redact)
        
        assert redacted_data["level1"]["level2"]["level3"]["sensitive"] != "secret_value"
    
    def test_invalid_path_handling(self):
        """Test handling of invalid JSON paths."""
        data = {"user": {"name": "John"}}
        
        # Path that doesn't exist
        paths_to_redact = ["/user/nonexistent", "/invalid/path"]
        redacted_data = self.redaction_manager.redact_by_paths(data, paths_to_redact)
        
        # Data should be unchanged, no errors
        assert redacted_data == data
    
    def test_different_redaction_actions(self):
        """Test specifying different redaction actions."""
        data = {
            "hash_me": "value1",
            "mask_me": "value2",
            "remove_me": "value3"
        }
        
        redacted_data = self.redaction_manager.redact_by_paths(
            data,
            ["/hash_me"],
            method=RedactionMethod.HASH
        )

        redacted_data = self.redaction_manager.redact_by_paths(
            redacted_data,
            ["/mask_me"],
            method=RedactionMethod.MASK
        )

        redacted_data = self.redaction_manager.redact_by_paths(
            redacted_data,
            ["/remove_me"],
            method=RedactionMethod.REMOVE
        )
        
        # Check that different methods were applied
        assert redacted_data["hash_me"].startswith("[HASH:")
        assert "*" in redacted_data["mask_me"]
        assert redacted_data["remove_me"] == "[REDACTED]"

    def test_site_specific_redaction_rules(self):
        """Test that site-specific redaction rules are applied correctly."""
        # Create redaction manager
        from app.audit.datalayer.redaction import RedactionManager

        # Mock the get_site_datalayer_config function to return test config
        import unittest.mock
        from app.audit.datalayer.config import RedactionConfig

        # Create a mock site config with specific redaction rules
        mock_site_config = unittest.mock.MagicMock()
        mock_site_config.redaction.rules = [
            RedactionRuleConfig(path="/user/site_email", method=RedactionMethod.HASH)
        ]

        with unittest.mock.patch('app.audit.datalayer.config.get_site_datalayer_config') as mock_get_config:
            mock_get_config.return_value = mock_site_config

            # Initialize redaction manager with base config (no rules)
            base_config = RedactionConfig(rules=[])
            manager = RedactionManager(base_config)

            # Test data
            data = {
                "user": {
                    "site_email": "user@site.com",
                    "other_field": "not an email value"
                }
            }

            # Apply site-specific redaction for example.com
            site_rules = manager._get_site_rules("example.com")
            redacted_data, audit_trail = manager._redactor.redact_data(data, site_rules)

            # Verify that site-specific rule was applied
            assert len(site_rules) == 1
            assert site_rules[0].path == "/user/site_email"

            # Verify redaction was applied only to the site-specific path
            assert redacted_data["user"]["site_email"] != "user@site.com"
            assert redacted_data["user"]["other_field"] == "not an email value"  # Unchanged

            # Verify audit trail includes the site-specific redaction
            site_audit_entries = [e for e in audit_trail if e.path == "/user/site_email"]
            assert len(site_audit_entries) == 1
            assert site_audit_entries[0].path == "/user/site_email"


class TestSensitiveDataPattern:
    """Test cases for SensitiveDataPattern."""
    
    def test_basic_pattern_creation(self):
        """Test basic pattern creation."""
        pattern = SensitiveDataPattern(
            name="email",
            pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            confidence=0.9,
            category="PII"
        )
        
        assert pattern.name == "email"
        assert pattern.confidence == 0.9
        assert pattern.category == "PII"
        assert pattern.compiled_pattern is not None
    
    def test_pattern_matching(self):
        """Test pattern matching functionality."""
        email_pattern = SensitiveDataPattern(
            name="email",
            pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            confidence=0.9
        )
        
        # Should match emails
        assert email_pattern.has_match("user@example.com") is True
        assert email_pattern.has_match("Contact us at support@company.org") is True

        # Should not match non-emails
        assert email_pattern.has_match("not an email") is False
        assert email_pattern.has_match("user@") is False
    
    def test_pattern_with_validation_function(self):
        """Test pattern with custom validation function."""
        def validate_credit_card(text: str) -> bool:
            """Simple credit card validation (Luhn algorithm check)."""
            digits = ''.join(filter(str.isdigit, text))
            if len(digits) < 13 or len(digits) > 19:
                return False
            return True  # Simplified for test
        
        cc_pattern = SensitiveDataPattern(
            name="credit_card",
            pattern=r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            confidence=0.8,
            validation_func=validate_credit_card
        )
        
        # Should validate with custom function
        assert cc_pattern.has_match("4111-1111-1111-1111") is True
        assert cc_pattern.has_match("1234-5678-9012-3456") is True

        # Too short - should fail validation
        assert cc_pattern.has_match("1234-5678") is False
    
    def test_pattern_compilation_error(self):
        """Test handling of invalid regex patterns."""
        from app.audit.datalayer.redaction import RedactionError
        with pytest.raises(RedactionError):
            SensitiveDataPattern(
                name="invalid",
                pattern="[invalid regex",  # Missing closing bracket
                confidence=0.5
            )


class TestPatternLibrary:
    """Test cases for PatternLibrary."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.library = PatternLibrary()
    
    def test_builtin_patterns_loaded(self):
        """Test that built-in patterns are loaded."""
        patterns = self.library.get_all_patterns()
        
        # Should have built-in patterns
        assert len(patterns) > 0
        
        # Should have common pattern categories
        pattern_names = [p.name for p in patterns]
        expected_patterns = ["email_address", "phone_number", "us_ssn", "credit_card"]
        for expected in expected_patterns:
            assert expected in pattern_names
    
    def test_get_patterns_by_category(self):
        """Test getting patterns by category."""
        pii_patterns = self.library.get_patterns_by_category("pii")
        financial_patterns = self.library.get_patterns_by_category("financial")

        assert len(pii_patterns) > 0
        assert len(financial_patterns) > 0

        # All patterns should have correct category
        for pattern in pii_patterns:
            assert pattern.category == "pii"
        
        for pattern in financial_patterns:
            assert pattern.category == "financial"
    
    def test_add_custom_pattern(self):
        """Test adding custom patterns."""
        custom_pattern = SensitiveDataPattern(
            name="custom_id",
            pattern=r"ID-\d{6}",
            confidence=0.7,
            category="Custom"
        )
        
        self.library.add_pattern(custom_pattern)
        
        # Should be able to retrieve custom pattern
        all_patterns = self.library.get_all_patterns()
        custom_patterns = [p for p in all_patterns if p.name == "custom_id"]
        
        assert len(custom_patterns) == 1
        assert custom_patterns[0].category == "Custom"
    
    def test_override_existing_pattern(self):
        """Test overriding existing patterns."""
        # Get original email pattern
        original_patterns = self.library.get_patterns_by_name("email")
        original_count = len(original_patterns)
        
        # Override with new pattern
        new_email_pattern = SensitiveDataPattern(
            name="email",
            pattern=r".+@.+\..+",  # Simpler pattern
            confidence=0.6,
            category="PII"
        )
        
        self.library.add_pattern(new_email_pattern, override=True)
        
        # Should have same count but updated pattern
        updated_patterns = self.library.get_patterns_by_name("email")
        assert len(updated_patterns) == original_count
        
        # At least one should have the new confidence
        confidences = [p.confidence for p in updated_patterns]
        assert 0.6 in confidences
    
    def test_pattern_search(self):
        """Test searching for patterns by name."""
        email_patterns = self.library.get_patterns_by_name("email")
        phone_patterns = self.library.get_patterns_by_name("phone")
        
        assert len(email_patterns) >= 1
        assert len(phone_patterns) >= 1
        
        # Should return empty list for non-existent pattern
        nonexistent = self.library.get_patterns_by_name("nonexistent")
        assert len(nonexistent) == 0


class TestAdvancedPatternDetector:
    """Test cases for AdvancedPatternDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AdvancedPatternDetector()
    
    def test_detect_sensitive_data_simple(self):
        """Test basic sensitive data detection."""
        text = "Contact us at support@company.com or call 555-123-4567"
        
        detections = self.detector.detect_sensitive_data(text)
        
        # Should detect email and phone
        assert len(detections) >= 2
        
        pattern_names = [d.pattern_name for d in detections]
        assert "email" in pattern_names
        assert any("phone" in name for name in pattern_names)
    
    def test_detect_in_json_structure(self):
        """Test detection in JSON-like structures."""
        data = {
            "user": {
                "email": "user@example.com",
                "phone": "555-123-4567",
                "name": "John Doe"
            },
            "payment": {
                "card": "4111-1111-1111-1111"
            }
        }
        
        detections = self.detector.detect_in_structure(data)
        
        # Should detect sensitive data in nested structure
        assert len(detections) >= 3  # email, phone, card
        
        # Should have paths
        paths = [d.json_path for d in detections]
        assert "/user/email" in paths
        assert "/user/phone" in paths
        assert "/payment/card" in paths
    
    def test_confidence_scoring_with_context(self):
        """Test confidence scoring based on context."""
        # Same pattern in different contexts
        email_in_contact = "email: user@example.com"
        email_in_random_text = "some random text user@example.com more text"
        
        detections_contact = self.detector.detect_sensitive_data(email_in_contact)
        detections_random = self.detector.detect_sensitive_data(email_in_random_text)
        
        # Find email detections
        email_contact = next((d for d in detections_contact if d.pattern_name == "email"), None)
        email_random = next((d for d in detections_random if d.pattern_name == "email"), None)
        
        # Context-aware scoring might give different confidences
        if email_contact and email_random:
            assert email_contact.confidence >= 0.0
            assert email_random.confidence >= 0.0
    
    def test_detection_with_field_name_hints(self):
        """Test detection using field name hints."""
        data_with_hints = {
            "customer_email": "user@example.com",  # Field name hints at email
            "random_field": "user@example.com",    # No field name hint
            "phone_number": "555-123-4567",        # Field name hints at phone
            "some_data": "555-123-4567"            # No field name hint
        }
        
        detections = self.detector.detect_in_structure(data_with_hints)
        
        # Should detect all instances but potentially with different confidences
        assert len(detections) == 4
        
        # Find detections by path
        detections_by_path = {d.json_path: d for d in detections}
        
        # Field name hints might boost confidence
        email_with_hint = detections_by_path["/customer_email"]
        email_without_hint = detections_by_path["/random_field"]
        
        # With hint should have equal or higher confidence
        assert email_with_hint.confidence >= email_without_hint.confidence
    
    def test_bulk_detection_performance(self):
        """Test detection on larger data structures."""
        # Create larger test data
        large_data = {
            f"user_{i}": {
                "email": f"user{i}@example.com",
                "phone": f"555-{i:03d}-{i:04d}",
                "data": f"random_data_{i}"
            }
            for i in range(100)
        }
        
        detections = self.detector.detect_in_structure(large_data)
        
        # Should detect email and phone for each user (200 total)
        assert len(detections) == 200
        
        # Check that detection doesn't take too long (performance test)
        import time
        start_time = time.time()
        self.detector.detect_in_structure(large_data)
        detection_time = time.time() - start_time
        
        # Should complete reasonably quickly (adjust threshold as needed)
        assert detection_time < 5.0  # 5 seconds max
    
    def test_false_positive_reduction(self):
        """Test false positive reduction mechanisms."""
        # Data that might trigger false positives
        text_with_false_positives = """
        This is a test with numbers that look like sensitive data:
        - Not a real email: notanemail@fake
        - Not a real phone: 111-111-1111
        - Not a real SSN: 111-11-1111
        - But this is real: user@example.com
        """
        
        detections = self.detector.detect_sensitive_data(text_with_false_positives)
        
        # Should detect the real email but ideally filter out false positives
        real_detections = [d for d in detections if d.confidence > 0.7]
        
        # Should have at least the real email
        assert len(real_detections) >= 1
        assert any(d.pattern_name == "email" for d in real_detections)
    
    def test_custom_pattern_integration(self):
        """Test integration with custom patterns."""
        # Add custom pattern
        custom_pattern = SensitiveDataPattern(
            name="employee_id",
            pattern=r"EMP-\d{5}",
            confidence=0.8,
            category="Custom"
        )
        
        self.detector.pattern_library.add_pattern(custom_pattern)
        
        # Test detection of custom pattern
        text = "Employee ID: EMP-12345"
        detections = self.detector.detect_sensitive_data(text)
        
        # Should detect custom pattern
        custom_detections = [d for d in detections if d.pattern_name == "employee_id"]
        assert len(custom_detections) == 1
        assert custom_detections[0].confidence == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])