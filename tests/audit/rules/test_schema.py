"""Unit tests for rule schema validation."""

import pytest
from pathlib import Path
import yaml
import json

from app.audit.rules.schema import (
    SchemaVersion,
    ValidationResult,
    get_schema_v0_1,
    get_schema_by_version,
    validate_rules_config,
    validate_rules_yaml,
    validate_rules_file,
    generate_example_config
)


class TestSchemaVersion:
    """Test SchemaVersion enum."""
    
    def test_schema_version_values(self):
        """Test schema version enum values."""
        assert SchemaVersion.V0_1 == "0.1"
    
    def test_schema_version_default(self):
        """Test default schema version."""
        # Should be the latest version
        assert SchemaVersion.V0_1 == "0.1"


class TestValidationResult:
    """Test ValidationResult model."""
    
    def test_validation_result_valid(self):
        """Test valid validation result."""
        result = ValidationResult(
            valid=True,
            errors=[],
            warnings=[]
        )
        
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.schema_version is None
    
    def test_validation_result_invalid(self):
        """Test invalid validation result."""
        errors = ["Missing required field: name"]
        warnings = ["Deprecated field used: old_param"]
        
        result = ValidationResult(
            valid=False,
            errors=errors,
            warnings=warnings,
            schema_version=SchemaVersion.V0_1
        )
        
        assert result.valid is False
        assert result.errors == errors
        assert result.warnings == warnings
        assert result.schema_version == SchemaVersion.V0_1


class TestSchemaRetrieval:
    """Test schema retrieval functions."""
    
    def test_get_schema_v0_1(self):
        """Test getting v0.1 schema."""
        schema = get_schema_v0_1()
        
        assert isinstance(schema, dict)
        assert "$schema" in schema
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        
        # Check for expected top-level properties
        properties = schema["properties"]
        assert "version" in properties
        assert "rules" in properties
    
    def test_get_schema_by_version(self):
        """Test getting schema by version."""
        # Valid version
        schema = get_schema_by_version(SchemaVersion.V0_1)
        assert isinstance(schema, dict)
        assert "properties" in schema
        
        # Same as direct v0.1 call
        direct_schema = get_schema_v0_1()
        assert schema == direct_schema
    
    def test_get_schema_invalid_version(self):
        """Test getting schema with invalid version."""
        with pytest.raises(ValueError, match="Unsupported schema version"):
            get_schema_by_version("999.999")  # type: ignore


class TestConfigValidation:
    """Test configuration validation functions."""
    
    def test_validate_valid_config(self):
        """Test validating a valid configuration."""
        valid_config = {
            "version": "0.1",
            "rules": [
                {
                    "id": "test_rule",
                    "name": "Test Rule",
                    "description": "A test rule",
                    "severity": "warning",
                    "check": {
                        "type": "request_present",
                        "parameters": {
                            "url_pattern": "analytics.js"
                        }
                    }
                }
            ]
        }
        
        result = validate_rules_config(valid_config)
        
        assert result.valid is True
        assert len(result.errors) == 0
        assert result.schema_version == SchemaVersion.V0_1
    
    def test_validate_missing_required_fields(self):
        """Test validation with missing required fields."""
        invalid_config = {
            "version": "0.1",
            "rules": [
                {
                    "id": "incomplete_rule",
                    # Missing name, description, severity, check
                }
            ]
        }
        
        result = validate_rules_config(invalid_config)
        
        assert result.valid is False
        assert len(result.errors) > 0
        # Should have errors for missing required fields
        error_text = " ".join(result.errors)
        assert "name" in error_text or "required" in error_text.lower()
    
    def test_validate_invalid_severity(self):
        """Test validation with invalid severity."""
        invalid_config = {
            "version": "0.1",
            "rules": [
                {
                    "id": "bad_severity_rule",
                    "name": "Bad Severity Rule",
                    "description": "Rule with invalid severity",
                    "severity": "invalid_severity",
                    "check": {
                        "type": "request_present",
                        "parameters": {}
                    }
                }
            ]
        }
        
        result = validate_rules_config(invalid_config)
        
        assert result.valid is False
        assert len(result.errors) > 0
    
    def test_validate_invalid_check_type(self):
        """Test validation with invalid check type."""
        invalid_config = {
            "version": "0.1", 
            "rules": [
                {
                    "id": "bad_check_rule",
                    "name": "Bad Check Rule",
                    "description": "Rule with invalid check type",
                    "severity": "warning",
                    "check": {
                        "type": "invalid_check_type",
                        "parameters": {}
                    }
                }
            ]
        }
        
        result = validate_rules_config(invalid_config)
        
        assert result.valid is False
        assert len(result.errors) > 0
    
    def test_validate_missing_version(self):
        """Test validation with missing version."""
        config_no_version = {
            "rules": [
                {
                    "id": "test_rule",
                    "name": "Test Rule", 
                    "description": "Test",
                    "severity": "info",
                    "check": {
                        "type": "request_present",
                        "parameters": {}
                    }
                }
            ]
        }
        
        result = validate_rules_config(config_no_version)
        
        # Should either be invalid or use default version
        if not result.valid:
            assert len(result.errors) > 0
    
    def test_validate_empty_rules(self):
        """Test validation with empty rules array."""
        empty_rules_config = {
            "version": "0.1",
            "rules": []
        }
        
        result = validate_rules_config(empty_rules_config)
        
        # Empty rules should be valid but might generate warnings
        assert result.valid is True
    
    def test_validate_complex_rule(self):
        """Test validation of complex rule with all fields."""
        complex_config = {
            "version": "0.1",
            "rules": [
                {
                    "id": "complex_rule",
                    "name": "Complex Test Rule",
                    "description": "A complex rule with all optional fields",
                    "severity": "critical",
                    "enabled": True,
                    "check": {
                        "type": "request_present",
                        "parameters": {
                            "url_pattern": "https://.*\\.googletagmanager\\.com/gtm\\.js\\?id=GTM-.*",
                            "method": "GET",
                            "timeout": 5000
                        },
                        "timeout_seconds": 30,
                        "retry_count": 2,
                        "enabled": True
                    },
                    "applies_to": {
                        "urls": ["https://example.com/*", "https://shop.example.com/*"],
                        "environments": ["production", "staging"],
                        "scope": "page"
                    },
                    "alert": {
                        "enabled": True,
                        "channels": ["webhook", "email"],
                        "webhook_url": "https://hooks.slack.com/...",
                        "email_recipients": ["admin@example.com"],
                        "template": {
                            "title": "Rule Failed: {{rule_name}}",
                            "message": "Rule {{rule_id}} failed with message: {{failure_message}}"
                        }
                    }
                }
            ]
        }
        
        result = validate_rules_config(complex_config)
        
        assert result.valid is True
        assert len(result.errors) == 0


class TestYamlValidation:
    """Test YAML validation functions."""
    
    def test_validate_valid_yaml_string(self):
        """Test validating valid YAML string."""
        yaml_content = """
version: "0.1"
rules:
  - id: yaml_test_rule
    name: YAML Test Rule
    description: Testing YAML validation
    severity: info
    check:
      type: request_present
      parameters:
        url_pattern: "test.js"
"""
        
        result = validate_rules_yaml(yaml_content)
        
        assert result.valid is True
        assert len(result.errors) == 0
    
    def test_validate_invalid_yaml_syntax(self):
        """Test validating YAML with syntax errors."""
        invalid_yaml = """
version: "0.1"
rules:
  - id: broken_rule
    name: Broken Rule
    description: YAML syntax error test
    severity: info
    check:
      type: presence
      parameters:
        url_pattern: "test.js"
        # Missing closing quote and indentation error
        bad_param: "unclosed string
      bad_indent: value
"""
        
        result = validate_rules_yaml(invalid_yaml)
        
        assert result.valid is False
        assert len(result.errors) > 0
        # Should mention YAML parsing error
        error_text = " ".join(result.errors).lower()
        assert "yaml" in error_text or "parse" in error_text
    
    def test_validate_yaml_with_schema_errors(self):
        """Test YAML that parses but fails schema validation."""
        schema_invalid_yaml = """
version: "0.1"
rules:
  - id: schema_invalid_rule
    # Missing required fields
    check:
      type: invalid_type
"""
        
        result = validate_rules_yaml(schema_invalid_yaml)
        
        assert result.valid is False
        assert len(result.errors) > 0


class TestFileValidation:
    """Test file validation functions."""
    
    def test_validate_nonexistent_file(self):
        """Test validating a file that doesn't exist."""
        nonexistent_path = Path("/nonexistent/path/to/rules.yaml")
        
        result = validate_rules_file(nonexistent_path)
        
        assert result.valid is False
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower() or "does not exist" in result.errors[0].lower()
    
    @pytest.fixture
    def temp_valid_rules_file(self, tmp_path):
        """Create a temporary valid rules file."""
        rules_content = {
            "version": "0.1",
            "rules": [
                {
                    "id": "temp_rule",
                    "name": "Temporary Rule",
                    "description": "Temporary test rule",
                    "severity": "warning",
                    "check": {
                        "type": "request_present",
                        "parameters": {
                            "url_pattern": "temp.js"
                        }
                    }
                }
            ]
        }
        
        temp_file = tmp_path / "valid_rules.yaml"
        with open(temp_file, 'w') as f:
            yaml.dump(rules_content, f)
        
        return temp_file
    
    @pytest.fixture
    def temp_invalid_rules_file(self, tmp_path):
        """Create a temporary invalid rules file."""
        temp_file = tmp_path / "invalid_rules.yaml"
        with open(temp_file, 'w') as f:
            f.write("invalid: yaml: content: [unclosed")
        
        return temp_file
    
    def test_validate_valid_file(self, temp_valid_rules_file):
        """Test validating a valid file."""
        result = validate_rules_file(temp_valid_rules_file)
        
        assert result.valid is True
        assert len(result.errors) == 0
    
    def test_validate_invalid_file(self, temp_invalid_rules_file):
        """Test validating an invalid file."""
        result = validate_rules_file(temp_invalid_rules_file)
        
        assert result.valid is False
        assert len(result.errors) > 0


class TestExampleGeneration:
    """Test example configuration generation."""
    
    def test_generate_example_config(self):
        """Test generating example configuration."""
        example = generate_example_config()
        
        assert isinstance(example, dict)
        assert "version" in example
        assert "rules" in example
        assert isinstance(example["rules"], list)
        assert len(example["rules"]) > 0
        
        # Validate the generated example
        result = validate_rules_config(example)
        assert result.valid is True
    
    def test_example_config_structure(self):
        """Test structure of generated example config."""
        example = generate_example_config()
        
        # Should have at least one rule
        rules = example["rules"]
        assert len(rules) >= 1
        
        # First rule should have required fields
        first_rule = rules[0]
        required_fields = ["id", "name", "description", "severity", "check"]
        for field in required_fields:
            assert field in first_rule
        
        # Check structure should have type and parameters
        check = first_rule["check"]
        assert "type" in check
        assert "parameters" in check
    
    def test_example_config_serializable(self):
        """Test that example config is JSON/YAML serializable."""
        example = generate_example_config()
        
        # Should be JSON serializable
        json_str = json.dumps(example)
        assert len(json_str) > 0
        
        # Should be YAML serializable
        yaml_str = yaml.dump(example)
        assert len(yaml_str) > 0
        
        # Round trip should work
        json_parsed = json.loads(json_str)
        assert json_parsed == example
        
        yaml_parsed = yaml.safe_load(yaml_str)
        assert yaml_parsed == example


class TestSchemaEdgeCases:
    """Test schema validation edge cases."""
    
    def test_validate_none_config(self):
        """Test validating None configuration."""
        result = validate_rules_config(None)  # type: ignore
        
        assert result.valid is False
        assert len(result.errors) > 0
    
    def test_validate_empty_config(self):
        """Test validating empty configuration."""
        result = validate_rules_config({})
        
        assert result.valid is False
        assert len(result.errors) > 0
    
    def test_validate_non_dict_config(self):
        """Test validating non-dictionary configuration."""
        result = validate_rules_config("not a dict")  # type: ignore
        
        assert result.valid is False
        assert len(result.errors) > 0
    
    def test_validate_rules_not_list(self):
        """Test validation when rules is not a list."""
        config = {
            "version": "0.1",
            "rules": "not a list"
        }
        
        result = validate_rules_config(config)
        
        assert result.valid is False
        assert len(result.errors) > 0
    
    def test_additional_properties_handling(self):
        """Test handling of additional properties not in schema."""
        config_with_extra = {
            "version": "0.1",
            "extra_field": "should_be_ignored_or_warned",
            "rules": [
                {
                    "id": "test_rule",
                    "name": "Test Rule",
                    "description": "Test",
                    "severity": "info",
                    "check": {
                        "type": "request_present",
                        "parameters": {
                            "url_pattern": "test.js"
                        }
                    },
                    "custom_field": "extra_data"
                }
            ]
        }
        
        result = validate_rules_config(config_with_extra)
        
        # Should still be valid (additional properties allowed)
        # but might have warnings
        assert result.valid is True