"""Unit tests for DataLayer configuration."""

import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.audit.datalayer.config import (
    DataLayerConfig,
    RedactionConfig,
    SchemaConfig,
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
    
    def test_invalid_redaction_action(self):
        """Test invalid redaction action."""
        with pytest.raises(ValueError):
            RedactionConfig(default_action="INVALID_ACTION")


class TestValidationConfig:
    """Test cases for ValidationConfig model."""
    
    def test_default_validation_config(self):
        """Test default validation configuration."""
        config = ValidationConfig()
        
        assert config.enabled is True
        assert config.schema_path is None
        assert config.strict_mode is False
    
    def test_validation_config_with_schema(self):
        """Test validation configuration with schema."""
        config = ValidationConfig(
            enabled=True,
            schema_path="schemas/datalayer.json",
            strict_mode=True
        )
        
        assert config.schema_path == "schemas/datalayer.json"
        assert config.strict_mode is True


class TestConfigurationLoader:
    """Test cases for ConfigurationLoader."""
    
    def test_load_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "enabled": True,
            "capture_timeout": 15.0,
            "max_depth": 100,
            "redaction": {
                "enabled": True,
                "default_action": "MASK"
            }
        }
        
        loader = ConfigurationLoader()
        config = loader.load_from_dict(config_dict)
        
        assert config.enabled is True
        assert config.capture_timeout == 15.0
        assert config.redaction.default_action == "MASK"
    
    def test_load_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        config_data = {
            "enabled": True,
            "capture_timeout": 12.0,
            "validation": {
                "enabled": True,
                "strict_mode": True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            loader = ConfigurationLoader()
            config = loader.load_from_file(temp_path)
            
            assert config.enabled is True
            assert config.capture_timeout == 12.0
            assert config.validation.strict_mode is True
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        loader = ConfigurationLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load_from_file(Path("nonexistent.yaml"))
    
    def test_load_invalid_yaml(self):
        """Test loading invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = Path(f.name)
        
        try:
            loader = ConfigurationLoader()
            with pytest.raises(ConfigurationError):
                loader.load_from_file(temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_load_with_site_overrides(self):
        """Test loading with site-specific overrides."""
        config_data = {
            "enabled": True,
            "capture_timeout": 10.0,
            "site_overrides": {
                "example.com": {
                    "capture_timeout": 20.0
                },
                "test.com": {
                    "max_depth": 25
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = Path(f.name)
        
        try:
            loader = ConfigurationLoader()
            config = loader.load_from_file(temp_path)
            
            # Test site-specific override
            example_config = config.override_for_site("example.com", 
                                                    config_data["site_overrides"]["example.com"])
            assert example_config.capture_timeout == 20.0
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestConfigurationValidator:
    """Test cases for ConfigurationValidator."""
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = DataLayerConfig(
            enabled=True,
            capture_timeout=10.0,
            max_depth=50
        )
        
        validator = ConfigurationValidator()
        result = validator.validate(config)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_invalid_timeout(self):
        """Test validation with invalid timeout."""
        config_dict = {
            "enabled": True,
            "capture_timeout": -5.0,  # Invalid negative timeout
            "max_depth": 50
        }
        
        validator = ConfigurationValidator()
        result = validator.validate_dict(config_dict)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("timeout" in error.lower() for error in result.errors)
    
    def test_validate_incompatible_settings(self):
        """Test validation with incompatible settings."""
        config_dict = {
            "enabled": False,  # Disabled
            "validation": {
                "enabled": True,  # But validation enabled
                "strict_mode": True
            }
        }
        
        validator = ConfigurationValidator()
        result = validator.validate_dict(config_dict)
        
        assert not result.is_valid
        # Should have warning about validation enabled when main config is disabled
    
    def test_validate_with_warnings(self):
        """Test validation that produces warnings."""
        config_dict = {
            "enabled": True,
            "capture_timeout": 30.0,  # Very high timeout - should warn
            "max_size": 10485760  # Very large size - should warn
        }
        
        validator = ConfigurationValidator()
        result = validator.validate_dict(config_dict)
        
        assert result.is_valid  # Still valid but with warnings
        assert len(result.warnings) > 0


class TestRuntimeConfigurationManager:
    """Test cases for RuntimeConfigurationManager."""
    
    def test_basic_runtime_manager(self):
        """Test basic runtime configuration management."""
        config = DataLayerConfig(capture_timeout=10.0)
        
        manager = RuntimeConfigurationManager(config)
        
        assert manager.get_current_config().capture_timeout == 10.0
        assert manager.get_version() == 1
    
    def test_update_configuration(self):
        """Test updating configuration at runtime."""
        initial_config = DataLayerConfig(capture_timeout=10.0)
        manager = RuntimeConfigurationManager(initial_config)
        
        # Update configuration
        update_dict = {"capture_timeout": 15.0}
        success = manager.update_configuration(update_dict)
        
        assert success
        assert manager.get_current_config().capture_timeout == 15.0
        assert manager.get_version() == 2
    
    def test_invalid_runtime_update(self):
        """Test invalid runtime configuration update."""
        initial_config = DataLayerConfig(capture_timeout=10.0)
        manager = RuntimeConfigurationManager(initial_config)
        
        # Try invalid update
        invalid_update = {"capture_timeout": -5.0}
        success = manager.update_configuration(invalid_update)
        
        assert not success
        # Configuration should remain unchanged
        assert manager.get_current_config().capture_timeout == 10.0
        assert manager.get_version() == 1
    
    def test_configuration_history(self):
        """Test configuration change history."""
        initial_config = DataLayerConfig(capture_timeout=10.0)
        manager = RuntimeConfigurationManager(initial_config)
        
        # Make several updates
        manager.update_configuration({"capture_timeout": 15.0})
        manager.update_configuration({"max_depth": 75})
        
        history = manager.get_change_history()
        
        assert len(history) >= 2
        assert any(change["changes"]["capture_timeout"] == 15.0 for change in history)


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
        config = manager.get_template("production")
        
        assert config.enabled is True
        # Production should have conservative timeouts
        assert config.capture_timeout <= 15.0
        assert config.redaction.enabled is True
    
    def test_get_development_template(self):
        """Test getting development configuration template."""
        manager = ConfigurationTemplateManager()
        config = manager.get_template("development")
        
        assert config.enabled is True
        # Development might have relaxed settings
        assert config.validation.strict_mode is False
    
    def test_get_nonexistent_template(self):
        """Test getting nonexistent template."""
        manager = ConfigurationTemplateManager()
        
        with pytest.raises(ConfigurationError):
            manager.get_template("nonexistent_template")
    
    def test_register_custom_template(self):
        """Test registering custom configuration template."""
        manager = ConfigurationTemplateManager()
        
        custom_config = DataLayerConfig(
            enabled=True,
            capture_timeout=25.0,
            max_depth=200
        )
        
        manager.register_template("custom", custom_config)
        
        # Should be able to retrieve the custom template
        retrieved = manager.get_template("custom")
        assert retrieved.capture_timeout == 25.0
        assert retrieved.max_depth == 200
    
    def test_template_validation(self):
        """Test template validation during registration."""
        manager = ConfigurationTemplateManager()
        
        # Try to register invalid template
        invalid_config_dict = {
            "enabled": True,
            "capture_timeout": -10.0  # Invalid
        }
        
        with pytest.raises(ValidationError):
            manager.register_template_from_dict("invalid", invalid_config_dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])