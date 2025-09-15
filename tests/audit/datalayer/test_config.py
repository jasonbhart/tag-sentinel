"""Unit tests for DataLayer configuration."""

import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import ValidationError
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.audit.datalayer.config import (
    DataLayerConfig,
    RedactionConfig,
    SchemaConfig,
    CaptureConfig,
    DataLayerConfigManager,
    ConfigurationValidator,
    RuntimeConfigurationManager,
    ConfigurationTemplateManager,
    DataLayerConfigurationError
)


class TestDataLayerConfig:
    """Test cases for DataLayerConfig model."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = DataLayerConfig()
        
        assert config.environment == "production"
        assert config.capture.enabled is True
        assert config.capture.object_name == "dataLayer"
        assert config.capture.max_depth == 6
        assert config.redaction.enabled is True
        assert config.validation.enabled is True
        assert config.aggregation.enabled is True
    
    def test_config_with_custom_values(self):
        """Test configuration with custom values."""
        from app.audit.datalayer.config import CaptureConfig
        
        redaction_config = RedactionConfig()
        validation_config = SchemaConfig(schema_path="custom/schema.json")
        capture_config = CaptureConfig(max_depth=15)
        
        config = DataLayerConfig(
            environment="development",
            capture=capture_config,
            redaction=redaction_config,
            validation=validation_config
        )
        
        assert config.environment == "development"
        assert config.capture.max_depth == 15
        assert config.validation.schema_path == "custom/schema.json"
    
    def test_config_validation(self):
        """Test configuration validation."""
        from app.audit.datalayer.config import CaptureConfig
        
        # Test invalid environment
        with pytest.raises(ValueError):
            DataLayerConfig(environment="invalid")
        
        # Test invalid max_depth
        with pytest.raises(ValueError):
            capture_config = CaptureConfig(max_depth=0)
            DataLayerConfig(capture=capture_config)
        
        # Test valid environment values
        for env in ['development', 'staging', 'production', 'test']:
            config = DataLayerConfig(environment=env)
            assert config.environment == env
    
    def test_config_site_override(self):
        """Test site-specific configuration override."""
        base_config = DataLayerConfig(
            site_overrides={
                "example.com": {
                    "environment": "development"
                }
            }
        )
        
        # Test getting site config
        site_config = base_config.get_site_config("example.com")
        assert site_config is not None
        
        # Test non-existent site returns base config
        default_site_config = base_config.get_site_config("nonexistent.com")
        assert default_site_config == base_config


class TestRedactionConfig:
    """Test cases for RedactionConfig model."""
    
    def test_default_redaction_config(self):
        """Test default redaction configuration."""
        config = RedactionConfig()
        
        assert config.enabled is True
        assert config.default_method.value == "hash"
        assert config.pattern_detection is True
        assert len(config.patterns) > 0  # Should have default patterns
        assert "email" in config.patterns
        assert len(config.rules) > 0  # Should have default rules
    
    def test_redaction_config_with_patterns(self):
        """Test redaction configuration with patterns."""
        from app.audit.datalayer.models import RedactionMethod
        
        patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}-\d{3}-\d{4}\b"
        }
        
        config = RedactionConfig(
            enabled=True,
            default_method=RedactionMethod.MASK,
            patterns=patterns
        )
        
        assert config.default_method == RedactionMethod.MASK
        assert len(config.patterns) == 2
        assert "email" in config.patterns
    
    def test_invalid_redaction_method(self):
        """Test invalid redaction method."""
        with pytest.raises(ValueError):
            RedactionConfig(default_method="INVALID_METHOD")


class TestSchemaConfig:
    """Test cases for SchemaConfig model."""

    def test_default_schema_config(self):
        """Test default schema configuration."""
        config = SchemaConfig()

        assert config.enabled is True
        assert config.schema_path is None
        assert config.cache_schemas is True

    def test_schema_config_with_schema(self):
        """Test schema configuration with schema."""
        config = SchemaConfig(
            enabled=True,
            schema_path="schemas/datalayer.json",
            cache_schemas=False
        )

        assert config.schema_path == "schemas/datalayer.json"
        assert config.cache_schemas is False


class TestDataLayerConfigManager:
    """Test cases for DataLayerConfigManager."""

    def test_load_from_file(self):
        """Test loading configuration from file."""
        config_data = {
            "environment": "development",
            "capture": {
                "enabled": True,
                "execution_timeout_ms": 15000,
                "max_depth": 10
            },
            "redaction": {
                "enabled": True,
                "default_method": "mask"
            }
        }

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            manager = DataLayerConfigManager()
            config = manager.load_config(temp_path)

            assert config.environment == "development"
            assert config.capture.enabled is True
            assert config.capture.execution_timeout_ms == 15000
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_load_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "environment": "production",
            "capture": {
                "execution_timeout_ms": 12000
            },
            "validation": {
                "enabled": True,
                "cache_schemas": True
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            manager = DataLayerConfigManager()
            config = manager.load_config(temp_path)

            assert config.environment == "production"
            assert config.capture.execution_timeout_ms == 12000
            assert config.validation.cache_schemas is True
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        manager = DataLayerConfigManager()

        # Should succeed and return default config when file doesn't exist
        config = manager.load_config(Path("nonexistent.yaml"))

        # Should have default values
        assert config.environment == "production"
        assert config.capture.enabled is True

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = Path(f.name)

        try:
            manager = DataLayerConfigManager()
            with pytest.raises((yaml.YAMLError, DataLayerConfigurationError)):
                manager.load_config(temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_load_with_site_overrides(self):
        """Test loading with site-specific overrides."""
        config_data = {
            "environment": "development",
            "capture": {
                "execution_timeout_ms": 10000
            },
            "site_overrides": {
                "example.com": {
                    "capture": {
                        "execution_timeout_ms": 20000
                    }
                },
                "test.com": {
                    "capture": {
                        "max_depth": 15
                    }
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)

        try:
            manager = DataLayerConfigManager()
            config = manager.load_config(temp_path)

            # Test site-specific override
            example_config = config.get_site_config("example.com")
            assert example_config.capture.execution_timeout_ms == 20000
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestConfigurationValidator:
    """Test cases for ConfigurationValidator."""
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = DataLayerConfig(
            environment="development"
        )

        validator = ConfigurationValidator()
        result = validator.validate_comprehensive(config)

        # Should have validation result structure
        assert "errors" in result
        assert "warnings" in result
        assert len(result["errors"]) == 0
    
    def test_validate_invalid_timeout(self):
        """Test validation with invalid timeout."""
        # Create config with timeout that will generate warnings
        config = DataLayerConfig(
            environment="development",
            capture=CaptureConfig(execution_timeout_ms=100)  # Minimum valid timeout
        )

        validator = ConfigurationValidator()
        result = validator.validate_comprehensive(config)

        # Should have warnings about very low timeout
        assert "warnings" in result
        assert len(result["warnings"]) > 0
    
    def test_validate_incompatible_settings(self):
        """Test validation with incompatible settings."""
        # Create config with schema validation enabled but no schema path
        config = DataLayerConfig(
            environment="production",
            validation=SchemaConfig(
                enabled=True,
                schema_path=None  # Schema validation enabled but no path
            )
        )

        validator = ConfigurationValidator()
        result = validator.validate_comprehensive(config)

        # Should have warning about schema validation enabled but no schema_path
        assert "warnings" in result
        assert len(result["warnings"]) > 0
    
    def test_validate_with_warnings(self):
        """Test validation that produces warnings."""
        # Create config that should generate warnings
        config = DataLayerConfig(
            environment="production",
            capture=CaptureConfig(
                execution_timeout_ms=30000,  # Very high timeout - should warn
                max_size_bytes=104857600  # Very large size - should warn
            )
        )

        validator = ConfigurationValidator()
        result = validator.validate_comprehensive(config)

        # Should have warnings about high timeout and large size
        assert "warnings" in result
        assert len(result["warnings"]) > 0


class TestRuntimeConfigurationManager:
    """Test cases for RuntimeConfigurationManager."""
    
    def test_basic_runtime_manager(self):
        """Test basic runtime configuration management."""
        # Create a config manager with a config
        config_manager = DataLayerConfigManager()

        # Create runtime manager
        runtime_manager = RuntimeConfigurationManager(config_manager)

        # Should be able to get current config
        current_config = runtime_manager.config_manager.get_config()
        assert current_config.environment == "production"  # Default value
    
    def test_update_configuration(self):
        """Test updating configuration at runtime."""
        # Create config manager
        config_manager = DataLayerConfigManager()
        runtime_manager = RuntimeConfigurationManager(config_manager)

        # Update configuration
        update_dict = {
            "capture": {
                "execution_timeout_ms": 15000
            }
        }
        result = runtime_manager.update_runtime_config(update_dict)

        # Should return result dict
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is True
    
    def test_invalid_runtime_update(self):
        """Test invalid runtime configuration update."""
        config_manager = DataLayerConfigManager()
        runtime_manager = RuntimeConfigurationManager(config_manager)

        # Try invalid update (negative timeout)
        invalid_update = {
            "capture": {
                "execution_timeout_ms": -5  # Invalid negative value
            }
        }
        result = runtime_manager.update_runtime_config(invalid_update)

        # Should return unsuccessful result
        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is False
    
    def test_configuration_history(self):
        """Test configuration change history."""
        config_manager = DataLayerConfigManager()
        runtime_manager = RuntimeConfigurationManager(config_manager)

        # Make several updates
        update1 = {
            "capture": {
                "execution_timeout_ms": 15000
            }
        }
        runtime_manager.update_runtime_config(update1)

        update2 = {
            "capture": {
                "max_depth": 15
            }
        }
        runtime_manager.update_runtime_config(update2)

        # Test that runtime overrides are tracked
        overrides = runtime_manager.get_runtime_overrides()
        assert isinstance(overrides, dict)


class TestConfigurationTemplateManager:
    """Test cases for ConfigurationTemplateManager."""
    
    def test_list_available_templates(self):
        """Test listing available configuration templates."""
        manager = ConfigurationTemplateManager()
        templates = manager.list_templates()
        
        # Should have at least the built-in templates
        expected_templates = ["production", "development", "testing", "minimal"]
        for template in expected_templates:
            assert template in templates
    
    def test_get_production_template(self):
        """Test getting production configuration template."""
        manager = ConfigurationTemplateManager()
        config_dict = manager.get_template("production")

        # Should return a dictionary
        assert isinstance(config_dict, dict)
        # Should have basic configuration sections
        assert "aggregation" in config_dict
        assert "redaction" in config_dict
    
    def test_get_development_template(self):
        """Test getting development configuration template."""
        manager = ConfigurationTemplateManager()
        config_dict = manager.get_template("development")

        # Should return a dictionary
        assert isinstance(config_dict, dict)
        # Should have basic configuration sections
        assert "aggregation" in config_dict
        assert "redaction" in config_dict
    
    def test_get_nonexistent_template(self):
        """Test getting nonexistent template."""
        manager = ConfigurationTemplateManager()
        
        with pytest.raises(ValueError):
            manager.get_template("nonexistent_template")
    
    def test_register_custom_template(self):
        """Test registering custom configuration template."""
        manager = ConfigurationTemplateManager()

        custom_template = {
            "capture": {
                "execution_timeout_ms": 25000,
                "max_depth": 15
            },
            "redaction": {
                "enabled": True
            }
        }

        manager.add_template("custom", custom_template)

        # Should be able to retrieve the custom template
        retrieved = manager.get_template("custom")
        assert retrieved["capture"]["execution_timeout_ms"] == 25000
        assert retrieved["capture"]["max_depth"] == 15
    
    def test_template_validation(self):
        """Test template validation when creating config from template."""
        manager = ConfigurationTemplateManager()

        # Add a template with invalid values
        invalid_template = {
            "capture": {
                "execution_timeout_ms": -10  # Invalid negative value
            }
        }

        manager.add_template("invalid", invalid_template)

        # Should fail when trying to create config from invalid template
        with pytest.raises(ValidationError):
            manager.create_config_from_template("invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])