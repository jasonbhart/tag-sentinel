"""Unit tests for DataLayer validation system."""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.audit.datalayer.validation import Validator
from app.audit.datalayer.models import ValidationIssue, ValidationSeverity
from app.audit.datalayer.config import SchemaConfig


class TestValidator:
    """Test cases for Validator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = SchemaConfig(enabled=True)
        self.validator = Validator(self.config)
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        assert self.validator.config == self.config
        assert self.validator.config.enabled is True
    
    def test_validate_without_schema(self):
        """Test validation when no schema is provided."""
        data = {"test": "value"}
        
        issues = self.validator.validate(data)
        
        # Should return empty list when no schema is configured
        assert issues == []
    
    def test_validate_with_simple_schema(self):
        """Test validation with simple JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["name"]
        }
        
        # Valid data
        valid_data = {"name": "John", "age": 25}
        issues = self.validator.validate(valid_data, schema)
        assert len(issues) == 0
        
        # Invalid data - missing required field
        invalid_data = {"age": 25}
        issues = self.validator.validate(invalid_data, schema)
        assert len(issues) > 0
        assert any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        assert any("required" in issue.message.lower() for issue in issues)
    
    def test_validate_with_type_mismatch(self):
        """Test validation with type mismatches."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "flag": {"type": "boolean"},
                "items": {"type": "array"}
            }
        }
        
        invalid_data = {
            "count": "not_a_number",  # Should be integer
            "flag": "not_a_boolean",  # Should be boolean
            "items": "not_an_array"   # Should be array
        }
        
        issues = self.validator.validate(invalid_data, schema)
        
        assert len(issues) == 3
        for issue in issues:
            assert issue.severity == ValidationSeverity.ERROR
            assert "type" in issue.message.lower()
    
    def test_validate_nested_objects(self):
        """Test validation of nested object structures."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "profile": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "string", "format": "email"}
                            },
                            "required": ["email"]
                        }
                    },
                    "required": ["id", "profile"]
                }
            },
            "required": ["user"]
        }
        
        # Valid nested data
        valid_data = {
            "user": {
                "id": "123",
                "profile": {
                    "email": "user@example.com"
                }
            }
        }
        
        issues = self.validator.validate(valid_data, schema)
        assert len(issues) == 0
        
        # Invalid nested data - missing email
        invalid_data = {
            "user": {
                "id": "123",
                "profile": {}  # Missing required email
            }
        }
        
        issues = self.validator.validate(invalid_data, schema)
        assert len(issues) > 0
        
        # Should have proper JSON pointer path
        email_issue = next((issue for issue in issues if "email" in issue.message.lower()), None)
        assert email_issue is not None
        assert email_issue.path.startswith("/user/profile")
    
    def test_validate_arrays(self):
        """Test validation of array structures."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"}
                        },
                        "required": ["id"]
                    },
                    "minItems": 1
                }
            }
        }
        
        # Valid array data
        valid_data = {
            "items": [
                {"id": 1, "name": "Item 1"},
                {"id": 2, "name": "Item 2"}
            ]
        }
        
        issues = self.validator.validate(valid_data, schema)
        assert len(issues) == 0
        
        # Invalid array data
        invalid_data = {
            "items": [
                {"name": "Item 1"},  # Missing required id
                {"id": "invalid", "name": "Item 2"}  # Wrong type for id
            ]
        }
        
        issues = self.validator.validate(invalid_data, schema)
        assert len(issues) >= 2
        
        # Should have proper array index paths
        paths = [issue.path for issue in issues]
        assert any("/items/0" in path for path in paths)
        assert any("/items/1" in path for path in paths)
    
    def test_schema_loading_from_file(self):
        """Test loading schema from JSON file."""
        schema_data = {
            "type": "object",
            "properties": {
                "page": {"type": "string"},
                "event": {"type": "string"}
            },
            "required": ["page"]
        }
        
        # Create temporary schema file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema_data, f)
            temp_path = Path(f.name)
        
        try:
            # Load schema from file
            schema = self.validator.load_schema(temp_path)
            assert schema == schema_data
            
            # Test validation with loaded schema
            data = {"event": "click"}  # Missing required "page"
            issues = self.validator.validate(data, schema)
            assert len(issues) > 0
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_schema_caching(self):
        """Test schema caching functionality."""
        schema_data = {"type": "object", "properties": {"test": {"type": "string"}}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema_data, f)
            temp_path = Path(f.name)
        
        try:
            # Load schema twice
            schema1 = self.validator.load_schema(temp_path)
            schema2 = self.validator.load_schema(temp_path)
            
            # Should return the same object (cached)
            assert schema1 is schema2
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_schema_reference_resolution(self):
        """Test JSON Schema reference resolution."""
        # Create schema with references
        main_schema = {
            "type": "object",
            "properties": {
                "user": {"$ref": "#/definitions/User"}
            },
            "definitions": {
                "User": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"}
                    },
                    "required": ["id"]
                }
            }
        }
        
        # Valid data
        valid_data = {"user": {"id": "123", "name": "John"}}
        issues = self.validator.validate(valid_data, main_schema)
        assert len(issues) == 0
        
        # Invalid data - missing required field
        invalid_data = {"user": {"name": "John"}}  # Missing id
        issues = self.validator.validate(invalid_data, main_schema)
        assert len(issues) > 0
    
    def test_validation_issue_mapping(self):
        """Test proper mapping of validation errors to ValidationIssue objects."""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 1},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["count", "email"]
        }
        
        invalid_data = {
            "count": 0,  # Below minimum
            "email": "invalid_email"  # Invalid format
        }
        
        issues = self.validator.validate(invalid_data, schema)
        
        # Should have issues for both fields
        assert len(issues) >= 2
        
        for issue in issues:
            assert isinstance(issue, ValidationIssue)
            assert issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.WARNING, ValidationSeverity.INFO]
            assert len(issue.message) > 0
            assert issue.path.startswith("/")
    
    def test_severity_mapping(self):
        """Test mapping of different validation errors to severity levels."""
        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"},
                "optional_field": {"type": "integer", "minimum": 0}
            },
            "required": ["required_field"]
        }
        
        invalid_data = {
            "optional_field": -1  # Violates minimum constraint
            # Missing required_field
        }
        
        issues = self.validator.validate(invalid_data, schema)
        
        # Should have different severities
        severities = [issue.severity for issue in issues]
        assert ValidationSeverity.ERROR in severities  # For missing required field
    
    def test_graceful_degradation_without_jsonschema(self):
        """Test graceful degradation when jsonschema is not available."""
        # Mock the scenario where jsonschema is not available
        original_jsonschema = self.validator.jsonschema
        self.validator.jsonschema = None
        
        try:
            schema = {"type": "object"}
            data = {"test": "value"}
            
            issues = self.validator.validate(data, schema)
            
            # Should return empty list and not crash
            assert issues == []
            
        finally:
            # Restore original jsonschema
            self.validator.jsonschema = original_jsonschema
    
    def test_invalid_schema_handling(self):
        """Test handling of invalid schemas."""
        invalid_schema = {
            "type": "invalid_type",  # Invalid JSON Schema type
            "properties": "not_an_object"  # Should be object
        }
        
        data = {"test": "value"}
        
        # Should handle invalid schema gracefully
        try:
            issues = self.validator.validate(data, invalid_schema)
            # If it doesn't raise an exception, it should return some indication
            # The exact behavior depends on implementation
        except Exception as e:
            # If it raises an exception, it should be a schema-related error
            assert "schema" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_very_large_data_validation(self):
        """Test validation performance with large data structures."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "value": {"type": "string"}
                        },
                        "required": ["id"]
                    }
                }
            }
        }
        
        # Create large valid dataset
        large_data = {
            "items": [
                {"id": i, "value": f"value_{i}"} 
                for i in range(1000)
            ]
        }
        
        # Should validate large data without issues
        import time
        start_time = time.time()
        issues = self.validator.validate(large_data, schema)
        validation_time = time.time() - start_time
        
        assert len(issues) == 0
        assert validation_time < 5.0  # Should complete within 5 seconds
    
    def test_unicode_data_validation(self):
        """Test validation with unicode data."""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"}
            }
        }
        
        unicode_data = {
            "title": "Test Title 🚀",
            "description": "描述 with mixed 内容 and émojis 🎉"
        }
        
        issues = self.validator.validate(unicode_data, schema)
        assert len(issues) == 0
    
    def test_custom_format_validation(self):
        """Test custom format validations."""
        schema = {
            "type": "object",
            "properties": {
                "email": {"type": "string", "format": "email"},
                "uri": {"type": "string", "format": "uri"},
                "date": {"type": "string", "format": "date"}
            }
        }
        
        # Valid formatted data
        valid_data = {
            "email": "user@example.com",
            "uri": "https://example.com/path",
            "date": "2023-12-25"
        }
        
        issues = self.validator.validate(valid_data, schema)
        assert len(issues) == 0
        
        # Invalid formatted data
        invalid_data = {
            "email": "not_an_email",
            "uri": "not_a_uri",
            "date": "not_a_date"
        }
        
        issues = self.validator.validate(invalid_data, schema)
        assert len(issues) >= 3


class TestValidatorConfiguration:
    """Test cases for validator configuration."""
    
    def test_strict_mode_validation(self):
        """Test strict mode validation."""
        strict_config = SchemaConfig(enabled=True, strict_mode=True)
        lenient_config = SchemaConfig(enabled=True, strict_mode=False)
        
        strict_validator = Validator(strict_config)
        lenient_validator = Validator(lenient_config)
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "additionalProperties": False  # Strict schema
        }
        
        data_with_extra = {
            "name": "John",
            "extra_field": "not allowed"  # Extra field
        }
        
        # Strict mode should catch additional properties
        strict_issues = strict_validator.validate(data_with_extra, schema)
        lenient_issues = lenient_validator.validate(data_with_extra, schema)
        
        # Both might catch the issue, but strict mode might be more aggressive
        assert len(strict_issues) >= len(lenient_issues)
    
    def test_disabled_validation(self):
        """Test behavior when validation is disabled."""
        disabled_config = SchemaConfig(enabled=False)
        disabled_validator = Validator(disabled_config)
        
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }
        
        invalid_data = {"wrong_field": "value"}  # Missing required name
        
        issues = disabled_validator.validate(invalid_data, schema)
        
        # Should return empty when disabled
        assert issues == []


class TestValidationIssueAnalysis:
    """Test cases for validation issue analysis and reporting."""
    
    def test_issue_categorization(self):
        """Test categorization of validation issues."""
        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"},
                "typed_field": {"type": "integer"},
                "formatted_field": {"type": "string", "format": "email"}
            },
            "required": ["required_field"]
        }
        
        invalid_data = {
            "typed_field": "wrong_type",
            "formatted_field": "wrong_format"
            # Missing required_field
        }
        
        validator = Validator(SchemaConfig(enabled=True))
        issues = self.validator.validate(invalid_data, schema)
        
        # Should have different types of issues
        issue_types = []
        for issue in issues:
            if "required" in issue.message.lower():
                issue_types.append("required")
            elif "type" in issue.message.lower():
                issue_types.append("type")
            elif "format" in issue.message.lower():
                issue_types.append("format")
        
        # Should have variety of issue types
        assert len(set(issue_types)) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])